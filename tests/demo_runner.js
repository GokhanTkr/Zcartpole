// Read a JSON request and compare browser dynamics with the Python tests.
const path = require("path");
const ZC = require(path.resolve(process.argv[2], "zcartpole.js"));

let input = "";
process.stdin.on("data", (d) => (input += d));
process.stdin.on("end", () => {
  const req = JSON.parse(input);
  const mdl = ZC.makeModel(req.model);
  const out = {};

  // One RK4 step with constant force.
  let st = { x: req.step.x, v: req.step.v, c: req.step.c, s: req.step.s, w: req.step.w };
  const s1 = ZC.step(mdl, st, req.step.u, req.step.h);
  out.step = { x: s1.x, v: s1.v, c: s1.c, s: s1.s, w: s1.w };

  // Closed-loop LQR: compute force, record it, then advance.
  st = ZC.makeState(0, req.loop.th0, null);
  const h = req.loop.dt, steps = req.loop.steps;
  const U = [];
  for (let k = 0; k <= steps; k++) {
    const u = ZC.lqrControl(mdl, st);
    U.push(u);
    if (k < steps) st = ZC.step(mdl, st, u, h);
  }
  out.loop = { x: st.x, v: st.v, c: st.c, s: st.s, w: st.w, U: U };

  // Open-loop motion with zero force.
  st = ZC.makeState(0, req.free.th0, req.free.w0);
  for (let k = 0; k < req.free.steps; k++) st = ZC.step(mdl, st, 0, req.free.dt);
  out.free = { x: st.x, v: st.v, c: st.c, s: st.s, w: st.w };

  process.stdout.write(JSON.stringify(out));
});
