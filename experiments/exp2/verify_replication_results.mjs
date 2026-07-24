#!/usr/bin/env bun
import { readFileSync } from "node:fs";
import { join } from "node:path";

const exp2 = import.meta.dir;
const cells = ["ar_fp16", "ar_ternary", "dllm_fp16", "dllm_ternary"];
const log = (seed, cell) => join(exp2, "logs", seed === 1 ? `${cell}.log` : `seed-${seed}_${cell}.log`);
const nll = (seed, cell) => {
  const values = [...readFileSync(log(seed, cell), "utf8").matchAll(/val\/nll' reached ([\d.]+).*best/g)];
  if (!values.length) throw new Error(`missing best val/nll: seed ${seed} ${cell}`);
  return Number(values.at(-1)[1]);
};
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
const sampleSd = values => Math.sqrt(values.reduce((sum, value) => sum + (value - mean(values)) ** 2, 0) / (values.length - 1));

const rows = [1, 2, 3].map(seed => {
  const arGap = Math.exp(nll(seed, "ar_ternary") - nll(seed, "ar_fp16"));
  const dllmGap = Math.exp(nll(seed, "dllm_ternary") - nll(seed, "dllm_fp16"));
  return { seed, arGap, dllmGap, r: dllmGap / arGap };
});
const logR = rows.map(({ r }) => Math.log(r));
const logMean = mean(logR);
const margin = 4.302652729 * sampleSd(logR) / Math.sqrt(logR.length); // two-sided 95%, df=2
const ci = [Math.exp(logMean - margin), Math.exp(logMean + margin)];

console.table(rows.map(({ seed, arGap, dllmGap, r }) => ({ seed, arGap, dllmGap, r })));
console.log(`geometric mean R=${Math.exp(logMean).toFixed(3)}; 95% t CI=[${ci.map(value => value.toFixed(3)).join(", ")}]`);
if (ci[1] > 1.25 || ci[1] < 0.8) throw new Error("unexpected preregistration verdict");
