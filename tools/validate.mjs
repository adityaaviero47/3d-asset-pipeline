import validator from 'gltf-validator';
import fs from 'fs';
const out = {};
for (const f of process.argv.slice(2)) {
  const r = await validator.validateBytes(new Uint8Array(fs.readFileSync(f)));
  out[f] = {errors: r.issues.numErrors, warnings: r.issues.numWarnings, infos: r.issues.numInfos,
            messages: r.issues.messages.filter(m => m.severity === 0).slice(0, 5).map(m => m.code + ': ' + m.message)};
}
console.log(JSON.stringify(out));
