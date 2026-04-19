use core::ptr;
use rknpu2::f16;

#[derive(Debug)]
pub struct PackedKv<const L: usize, const H: usize> {
    pub b: usize,     // 1
    pub t_max: usize, // cache length
    pub d: usize,     // head dim (64)
    pub k: Vec<f16>,  // len = b*l*h*t_max*d
    pub v: Vec<f16>,
    pub t: usize,
}

impl<const L: usize, const H: usize> PackedKv<L, H> {
    pub fn new(t_max: usize, d: usize) -> Self {
        let b = 1;
        let len = b * L * H * t_max * d;
        Self {
            b,
            t_max,
            d,
            k: vec![f16::from_f32(0.0); len],
            v: vec![f16::from_f32(0.0); len],
            t: 0,
        }
    }

    pub fn clear(&mut self) {
        self.k.fill(f16::from_f32(0.0));
        self.v.fill(f16::from_f32(0.0));
        self.t = 0;
    }

    #[inline]
    pub fn past_len(&self) -> usize {
        self.t
    }

    #[inline]
    pub fn offset(&self) -> usize {
        if self.past_len() <= self.t_max {
            0
        } else {
            self.past_len() - self.t_max
        }
    }

    pub fn bump(&mut self) {
        self.t += 1;
    }

    #[inline]
    pub fn base_offset(&self, layer: usize, head: usize, t: usize) -> usize {
        // (b=0 fixed)
        ((layer * H + head) * self.t_max + t) * self.d
    }

    /// Write a single time step for one LAYER from a present tensor slice.
    /// present_k/v shape: [H, T_grow, D] (contiguous row-major), we take the last time index (T_grow-1).
    pub fn write_step_from_present_one_layer(
        &mut self,
        layer: usize,
        pos: usize,        // where to write in [0, t_max-1]
        present_k: &[f16], // len = h * T_grow * d
        present_v: &[f16],
        t_grow: usize, // T_grow = pos+1 typically
    ) {
        debug_assert!(pos < self.t_max);
        let last_t = t_grow - 1;

        for head in 0..H {
            // src offsets inside present: [head, last_t, :]
            let src_off = (head * t_grow + last_t) * self.d;
            let src_k = &present_k[src_off..src_off + self.d];
            let src_v = &present_v[src_off..src_off + self.d];

            // dst offsets inside packed cache: [layer, head, pos, :]
            let dst_off = self.base_offset(layer, head, pos);
            let dst_k = &mut self.k[dst_off..dst_off + self.d];
            let dst_v = &mut self.v[dst_off..dst_off + self.d];

            // memcpy (safe copy since slices are contiguous)
            dst_k.copy_from_slice(src_k);
            dst_v.copy_from_slice(src_v);
        }
    }

    /// Write a single time step for ALL layers.
    /// `presents_k/l` is a Vec per layer, each &[f16] with shape [H, T_grow, D].
    pub fn write_step_all_layers(
        &mut self,
        pos: usize,
        presents_k: &[&[f16]],
        presents_v: &[&[f16]],
        t_grow: usize,
    ) {
        assert_eq!(presents_k.len(), L);
        assert_eq!(presents_v.len(), L);
        for l in 0..L {
            self.write_step_from_present_one_layer(l, pos, presents_k[l], presents_v[l], t_grow);
        }
    }

    /// Compact when cache is full: left shift by 1 along T (drop t=0), keep last slot free.
    /// This preserves chronological order required by your fully-static graph.
    pub fn compact_left_one(&mut self) {
        let row_elems = self.d;
        // For each (layer, head), memmove T-1 rows of D elements from t=1.. to t=0..
        for l in 0..L {
            for h in 0..H {
                // source begins at t=1
                let src_off = self.base_offset(l, h, 1);
                // dest begins at t=0
                let dst_off = self.base_offset(l, h, 0);
                let count_rows = self.t_max - 1;
                let count_elems = count_rows * row_elems;

                // SAFETY: regions may overlap; use ptr::copy for memmove semantics
                unsafe {
                    let src_ptr = self.k.as_ptr().add(src_off);
                    let dst_ptr = self.k.as_mut_ptr().add(dst_off);
                    ptr::copy(src_ptr, dst_ptr, count_elems);
                }
                unsafe {
                    let src_ptr = self.v.as_ptr().add(src_off);
                    let dst_ptr = self.v.as_mut_ptr().add(dst_off);
                    ptr::copy(src_ptr, dst_ptr, count_elems);
                }

                // zero the last slot (t = t_max-1)
                let tail_off = self.base_offset(l, h, self.t_max - 1);
                let tail_k = &mut self.k[tail_off..tail_off + row_elems];
                let tail_v = &mut self.v[tail_off..tail_off + row_elems];
                for x in tail_k.iter_mut() {
                    *x = f16::from_f32(0.0);
                }
                for x in tail_v.iter_mut() {
                    *x = f16::from_f32(0.0);
                }
            }
        }
    }

    /// Expose raw slices to hand into RKNN input binding.
    pub fn as_slices(&self) -> (&[f16], &[f16]) {
        (&self.k, &self.v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rknpu2::f16;

    // Small, easy-to-reason-about fixture
    const L: usize = 2; // layers
    const H: usize = 3; // heads
    const T: usize = 5; // t_max
    const D: usize = 4; // head dim

    fn almost_eq(a: &[f16], b: &[f16], eps: f32) {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch {} vs {}",
            a.len(),
            b.len()
        );
        for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
            let da = av.to_f32();
            let db = bv.to_f32();
            assert!(
                (da - db).abs() <= eps,
                "mismatch at {}: {} vs {} (|diff|={})",
                i,
                da,
                db,
                (da - db).abs()
            );
        }
    }

    /// Build a present tensor for a single layer with shape [H, T_grow, D] (row-major flattened).
    /// We fill it as present_k[h, t, d] = base + encoded(h,t,d).
    fn synth_present_for_layer(
        heads: usize,
        t_grow: usize,
        d: usize,
        base: f32,
    ) -> (Vec<f16>, Vec<f16>) {
        let mut k = Vec::with_capacity(heads * t_grow * d);
        let mut v = Vec::with_capacity(heads * t_grow * d);
        for h in 0..heads {
            for tt in 0..t_grow {
                for dd in 0..d {
                    k.push(f16::from_f32(
                        base + (h as f32) * 100.0 + (tt as f32) * 10.0 + (dd as f32),
                    ));
                    v.push(f16::from_f32(
                        base + 0.5 + (h as f32) * 100.0 + (tt as f32) * 10.0 + (dd as f32),
                    ));
                }
            }
        }
        (k, v)
    }

    #[test]
    fn new_and_lengths() {
        let pk = PackedKv::<L, H>::new(T, D);
        assert_eq!(pk.b, 1);
        assert_eq!(pk.t_max, T);
        assert_eq!(pk.d, D);

        let expected_len = 1 * L * H * T * D;
        assert_eq!(pk.k.len(), expected_len);
        assert_eq!(pk.v.len(), expected_len);

        let (ks, vs) = pk.as_slices();
        assert_eq!(ks.len(), expected_len);
        assert_eq!(vs.len(), expected_len);
    }

    #[test]
    fn base_offset_is_correct() {
        let pk = PackedKv::<L, H>::new(T, D);

        // Manually compute some expected offsets:
        // offset(l,h,t,d0) = (((l*H + h)*T + t)*D)
        let cases = [
            (0, 0, 0, 0, 0usize),
            (0, 0, 0, 1, 1),
            (0, 0, 0, 3, 3),
            (0, 0, 1, 0, D * 1),
            (0, 1, 0, 0, (1 * T) * D),
            (1, 0, 0, 0, ((1 * H + 0) * T) * D),
            (1, 2, 4, 3, (((1 * H + 2) * T + 4) * D) + 3),
        ];

        for &(l, h, t, d0, want) in &cases {
            let got = pk.base_offset(l, h, t) + d0;
            assert_eq!(got, want, "off(l={},h={},t={},d={})", l, h, t, d0);
        }
    }

    #[test]
    fn write_step_one_layer_writes_last_t() {
        let mut pk = PackedKv::<L, H>::new(T, D);

        // Prepare present for layer 1, t_grow=3 (so last_t = 2)
        let t_grow = 3usize;
        let (present_k, present_v) = synth_present_for_layer(H, t_grow, D, 10_000.0);

        // Write at pos=4 (arbitrary within 0..T-1)
        let pos = 4usize;
        pk.write_step_from_present_one_layer(1, pos, &present_k, &present_v, t_grow);

        // Verify: for each head, the written row at (l=1, h, t=pos, :) equals present[..., last_t, :]
        let last_t = t_grow - 1;
        for h in 0..H {
            // cache slice
            let off = pk.base_offset(1, h, pos);
            let cache_row_k = &pk.k[off..off + D];
            let cache_row_v = &pk.v[off..off + D];

            // source slice from present
            let src_off = (h * t_grow + last_t) * D;
            let src_k = &present_k[src_off..src_off + D];
            let src_v = &present_v[src_off..src_off + D];

            almost_eq(cache_row_k, src_k, 1e-3);
            almost_eq(cache_row_v, src_v, 1e-3);
        }
    }

    #[test]
    fn write_step_all_layers_copies_each_layer() {
        let mut pk = PackedKv::<L, H>::new(T, D);

        // Build per-layer present buffers with distinct bases so we can distinguish layers.
        let t_grow = 2usize; // last_t = 1
        let mut layers_k: Vec<Vec<f16>> = Vec::with_capacity(L);
        let mut layers_v: Vec<Vec<f16>> = Vec::with_capacity(L);
        for l in 0..L {
            let base = 20_000.0 + (l as f32) * 1000.0;
            let (k, v) = synth_present_for_layer(H, t_grow, D, base);
            layers_k.push(k);
            layers_v.push(v);
        }
        let refs_k: Vec<&[f16]> = layers_k.iter().map(|x| x.as_slice()).collect();
        let refs_v: Vec<&[f16]> = layers_v.iter().map(|x| x.as_slice()).collect();

        // Write all layers at pos=0
        pk.write_step_all_layers(0, &refs_k, &refs_v, t_grow);

        // Check each layer's head row equals last_t=1 from its own present
        let last_t = t_grow - 1;
        for l in 0..L {
            for h in 0..H {
                let off = pk.base_offset(l, h, 0);
                let cache_row_k = &pk.k[off..off + D];
                let cache_row_v = &pk.v[off..off + D];

                let src_off = (h * t_grow + last_t) * D;
                let src_k = &layers_k[l][src_off..src_off + D];
                let src_v = &layers_v[l][src_off..src_off + D];

                almost_eq(cache_row_k, src_k, 1e-3);
                almost_eq(cache_row_v, src_v, 1e-3);
            }
        }
    }

    #[test]
    fn compact_left_one_shifts_and_zeros_tail() {
        let mut pk = PackedKv::<L, H>::new(T, D);

        // Fill cache with a simple function of t only so shift is easy to assert.
        // k(l,h,t,d) = t as f32, v(l,h,t,d) = 1000.0 + t as f32
        for l in 0..L {
            for h in 0..H {
                for t in 0..T {
                    let off = pk.base_offset(l, h, t);
                    for d0 in 0..D {
                        pk.k[off + d0] = f16::from_f32(t as f32);
                        pk.v[off + d0] = f16::from_f32(1000.0 + t as f32);
                    }
                }
            }
        }

        pk.compact_left_one();

        // After compaction:
        // t=0 should equal old t=1
        // ...
        // t=T-2 should equal old t=T-1
        // t=T-1 should be zeros
        for l in 0..L {
            for h in 0..H {
                // check shifted rows
                for t_new in 0..(T - 1) {
                    let off_new = pk.base_offset(l, h, t_new);
                    let expect_t = t_new + 1; // old t
                    for d0 in 0..D {
                        let got_k = pk.k[off_new + d0].to_f32();
                        let got_v = pk.v[off_new + d0].to_f32();
                        assert!(
                            (got_k - expect_t as f32).abs() <= 1e-3,
                            "k mismatch at (l={},h={},t={},d={}) got {} expect {}",
                            l,
                            h,
                            t_new,
                            d0,
                            got_k,
                            expect_t
                        );
                        assert!(
                            (got_v - (1000.0 + expect_t as f32)).abs() <= 1e-3,
                            "v mismatch at (l={},h={},t={},d={}) got {} expect {}",
                            l,
                            h,
                            t_new,
                            d0,
                            got_v,
                            1000.0 + expect_t as f32
                        );
                    }
                }

                // last row zeroed
                let off_last = pk.base_offset(l, h, T - 1);
                for d0 in 0..D {
                    let got_k = pk.k[off_last + d0].to_f32();
                    let got_v = pk.v[off_last + d0].to_f32();
                    assert!(
                        got_k.abs() <= 1e-6,
                        "tail k not zero at (l={},h={},d={}) -> {}",
                        l,
                        h,
                        d0,
                        got_k
                    );
                    assert!(
                        got_v.abs() <= 1e-6,
                        "tail v not zero at (l={},h={},d={}) -> {}",
                        l,
                        h,
                        d0,
                        got_v
                    );
                }
            }
        }
    }

    #[test]
    fn as_slices_views_match_internal() {
        let mut pk = PackedKv::<L, H>::new(T, D);

        // Write a known pattern at a few positions
        let t_grow = 2usize;
        for l in 0..L {
            let (k, v) = synth_present_for_layer(H, t_grow, D, 30_000.0 + (l as f32) * 100.0);
            // write pos=1 for each layer
            pk.write_step_from_present_one_layer(l, 1, &k, &v, t_grow);
        }

        // as_slices should point to the same data
        let (ks, vs) = pk.as_slices();
        assert_eq!(ks.as_ptr(), pk.k.as_ptr());
        assert_eq!(vs.as_ptr(), pk.v.as_ptr());

        // Spot check a couple cells against direct indexing
        let sample = [
            (0usize, 0usize, 1usize, 2usize),
            (1usize, 2usize, 1usize, 3usize),
        ];
        for &(l, h, t, d0) in &sample {
            let off = pk.base_offset(l, h, t) + d0;
            assert!(
                (ks[off].to_f32() - pk.k[off].to_f32()).abs() <= 1e-6,
                "k view mismatch at {:?}",
                (l, h, t, d0)
            );
            assert!(
                (vs[off].to_f32() - pk.v[off].to_f32()).abs() <= 1e-6,
                "v view mismatch at {:?}",
                (l, h, t, d0)
            );
        }
    }

    #[test]
    fn full_window_fill_compact_and_append() {
        let mut pk = PackedKv::<L, H>::new(T, D);

        // Keep within fp16 range: worst-case s=T (5) -> ~26.5k
        let base_for =
            |l: usize, s: usize| -> f32 { 10_000.0 + (l as f32) * 500.0 + (s as f32) * 3_000.0 };

        // -------- 1) Fill the whole window with steps s = 0..T-1 --------
        for s in 0..T {
            let t_grow = s + 1; // last_t = s
            let mut layers_k: Vec<Vec<f16>> = Vec::with_capacity(L);
            let mut layers_v: Vec<Vec<f16>> = Vec::with_capacity(L);

            for l in 0..L {
                let (k, v) = synth_present_for_layer(H, t_grow, D, base_for(l, s));
                layers_k.push(k);
                layers_v.push(v);
            }

            let refs_k: Vec<&[f16]> = layers_k.iter().map(|x| x.as_slice()).collect();
            let refs_v: Vec<&[f16]> = layers_v.iter().map(|x| x.as_slice()).collect();
            pk.write_step_all_layers(s, &refs_k, &refs_v, t_grow);
        }

        // Verify every cell equals the "last_t" slice of its step s=t.
        for l in 0..L {
            for h in 0..H {
                for t in 0..T {
                    let off = pk.base_offset(l, h, t);
                    let src_off = (h * (t + 1) + t) * D; // last_t == t
                    let (src_k, src_v) = synth_present_for_layer(H, t + 1, D, base_for(l, t));
                    almost_eq(&pk.k[off..off + D], &src_k[src_off..src_off + D], 1e-3);
                    almost_eq(&pk.v[off..off + D], &src_v[src_off..src_off + D], 1e-3);
                }
            }
        }

        // -------- 2) Compact and verify left shift + zero tail --------
        let prev_k = pk.k.clone();
        let prev_v = pk.v.clone();
        pk.compact_left_one();

        for l in 0..L {
            for h in 0..H {
                for t_new in 0..(T - 1) {
                    let off_new = pk.base_offset(l, h, t_new);
                    let off_old = pk.base_offset(l, h, t_new + 1);
                    almost_eq(
                        &pk.k[off_new..off_new + D],
                        &prev_k[off_old..off_old + D],
                        1e-3,
                    );
                    almost_eq(
                        &pk.v[off_new..off_new + D],
                        &prev_v[off_old..off_old + D],
                        1e-3,
                    );
                }
                let off_tail = pk.base_offset(l, h, T - 1);
                for d0 in 0..D {
                    assert!(pk.k[off_tail + d0].to_f32().abs() <= 1e-6);
                    assert!(pk.v[off_tail + d0].to_f32().abs() <= 1e-6);
                }
            }
        }

        // -------- 3) Append one more step at pos = T-1 --------
        let s = T;
        let t_grow = s + 1; // last_t = s
        let pos = T - 1; // write into tail

        let mut layers_k: Vec<Vec<f16>> = Vec::with_capacity(L);
        let mut layers_v: Vec<Vec<f16>> = Vec::with_capacity(L);
        for l in 0..L {
            let (k, v) = synth_present_for_layer(H, t_grow, D, base_for(l, s));
            layers_k.push(k);
            layers_v.push(v);
        }
        let refs_k: Vec<&[f16]> = layers_k.iter().map(|x| x.as_slice()).collect();
        let refs_v: Vec<&[f16]> = layers_v.iter().map(|x| x.as_slice()).collect();
        pk.write_step_all_layers(pos, &refs_k, &refs_v, t_grow);

        for l in 0..L {
            for h in 0..H {
                let off = pk.base_offset(l, h, pos);
                let src_off = (h * t_grow + (t_grow - 1)) * D; // last_t
                almost_eq(
                    &pk.k[off..off + D],
                    &layers_k[l][src_off..src_off + D],
                    1e-3,
                );
                almost_eq(
                    &pk.v[off..off + D],
                    &layers_v[l][src_off..src_off + D],
                    1e-3,
                );
            }
        }
    }
}
