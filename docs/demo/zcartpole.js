/* Browser dynamics for the live LQR demo.
 * Python computes gain K in data.js. tests/test_demo_js.py checks agreement.
 * State: {x, v, c[], s[], w[]}; c/s are link cosine/sine, w is angular speed.
 */
(function (root) {
  "use strict";

  function makeModel(p) {
    const n = p.m.length;
    const S = new Array(n);
    let acc = 0;
    for (let i = n - 1; i >= 0; i--) { acc += p.m[i]; S[i] = acc; }
    return { n: n, m: p.m, l: p.l, M: p.M, g: p.g, uMax: p.u_max, S: S, K: p.K };
  }

  // Partial-pivot Gaussian elimination: A x = b
  function solve(A, b) {
    const N = b.length;
    const M = A.map(function (row, i) { return row.concat([b[i]]); });
    for (let k = 0; k < N; k++) {
      let piv = k;
      for (let i = k + 1; i < N; i++) if (Math.abs(M[i][k]) > Math.abs(M[piv][k])) piv = i;
      const tmp = M[k]; M[k] = M[piv]; M[piv] = tmp;
      for (let i = k + 1; i < N; i++) {
        const f = M[i][k] / M[k][k];
        for (let j = k; j <= N; j++) M[i][j] -= f * M[k][j];
      }
    }
    const x = new Array(N);
    for (let i = N - 1; i >= 0; i--) {
      let sum = M[i][N];
      for (let j = i + 1; j < N; j++) sum -= M[i][j] * x[j];
      x[i] = sum / M[i][i];
    }
    return x;
  }

  // [x_ddot, w1_dot, ..., wn_dot]; same indexing as core.py
  function accel(mdl, c, s, w, u) {
    const n = mdl.n, S = mdl.S, l = mdl.l, g = mdl.g, N = n + 1;
    const A = [];
    for (let i = 0; i < N; i++) A.push(new Array(N).fill(0));
    A[0][0] = mdl.M + S[0];
    for (let i = 1; i <= n; i++) {
      A[0][i] = A[i][0] = S[i - 1] * l[i - 1] * c[i - 1];
      A[i][i] = S[i - 1] * l[i - 1] * l[i - 1];
      for (let j = i + 1; j <= n; j++) {
        const rc = c[i - 1] * c[j - 1] + s[i - 1] * s[j - 1];     // Re(z_i conj z_j)
        A[i][j] = A[j][i] = S[j - 1] * l[i - 1] * l[j - 1] * rc;
      }
    }
    const F = new Array(N).fill(0);
    let F0 = u;
    for (let i = 0; i < n; i++) F0 += S[i] * l[i] * w[i] * w[i] * s[i];
    F[0] = F0;
    for (let i = 1; i <= n; i++) {
      let Fi = S[i - 1] * g * l[i - 1] * s[i - 1];
      for (let j = i + 1; j <= n; j++) {
        const rs = s[i - 1] * c[j - 1] - c[i - 1] * s[j - 1];     // Im(z_i conj z_j)
        Fi += -S[j - 1] * l[i - 1] * l[j - 1] * w[j - 1] * w[j - 1] * rs;
      }
      for (let j = 1; j < i; j++) {
        const rs = s[j - 1] * c[i - 1] - c[j - 1] * s[i - 1];     // Im(z_j conj z_i)
        Fi += S[i - 1] * l[j - 1] * l[i - 1] * w[j - 1] * w[j - 1] * rs;
      }
      F[i] = Fi;
    }
    return solve(A, F);
  }

  function makeState(x0, th0, w0) {
    const n = th0.length;
    return {
      x: x0, v: 0,
      c: th0.map(Math.cos), s: th0.map(Math.sin),
      w: w0 ? w0.slice() : new Array(n).fill(0)
    };
  }

  // One RK4 step with constant force; rotate z by exp(i phi)
  function step(mdl, st, u, h) {
    const n = mdl.n, dim = 2 + 2 * n;
    function f(y) {
      const c = new Array(n), s = new Array(n), w = new Array(n);
      for (let i = 0; i < n; i++) {
        const cp = Math.cos(y[2 + 2 * i]), sp = Math.sin(y[2 + 2 * i]);
        c[i] = st.c[i] * cp - st.s[i] * sp;
        s[i] = st.c[i] * sp + st.s[i] * cp;
        w[i] = y[3 + 2 * i];
      }
      const a = accel(mdl, c, s, w, u);
      const d = new Array(dim);
      d[0] = y[1]; d[1] = a[0];
      for (let i = 0; i < n; i++) { d[2 + 2 * i] = w[i]; d[3 + 2 * i] = a[i + 1]; }
      return d;
    }
    function axpy(y, k, a) { return y.map(function (yi, i) { return yi + a * k[i]; }); }
    const y0 = new Array(dim).fill(0);
    y0[0] = st.x; y0[1] = st.v;
    for (let i = 0; i < n; i++) y0[3 + 2 * i] = st.w[i];
    const k1 = f(y0);
    const k2 = f(axpy(y0, k1, 0.5 * h));
    const k3 = f(axpy(y0, k2, 0.5 * h));
    const k4 = f(axpy(y0, k3, h));
    const y = y0.map(function (yi, i) {
      return yi + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    });
    const c = new Array(n), s = new Array(n), w = new Array(n);
    for (let i = 0; i < n; i++) {
      const cp = Math.cos(y[2 + 2 * i]), sp = Math.sin(y[2 + 2 * i]);
      c[i] = st.c[i] * cp - st.s[i] * sp;
      s[i] = st.c[i] * sp + st.s[i] * cp;
      w[i] = y[3 + 2 * i];
    }
    return { x: y[0], v: y[1], c: c, s: s, w: w };
  }

  // Local error e = [x, Arg z_1..n, v, w_1..n], upright z = 1
  function error(mdl, st) {
    const e = [st.x];
    for (let i = 0; i < mdl.n; i++) e.push(Math.atan2(st.s[i], st.c[i]));
    e.push(st.v);
    for (let i = 0; i < mdl.n; i++) e.push(st.w[i]);
    return e;
  }

  function clip(u, lim) { return Math.max(-lim, Math.min(lim, u)); }

  function lqrControl(mdl, st) {
    const e = error(mdl, st);
    let u = 0;
    for (let k = 0; k < e.length; k++) u -= mdl.K[k] * e[k];
    return clip(u, mdl.uMax);
  }

  const api = {
    makeModel: makeModel, makeState: makeState, accel: accel, step: step,
    error: error, lqrControl: lqrControl, clip: clip
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ZC = api;
})(typeof window !== "undefined" ? window : this);
