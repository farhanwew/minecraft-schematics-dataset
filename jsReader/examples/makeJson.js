// Generates viewer/public/all_small.json from schematics_0.tfrecords
// Keeps only small schematics (W*H*L < 200000) for browser rendering

const nbt = require('prismarine-nbt')
const promisify = require('util').promisify
const parseNbt = promisify(nbt.parse)
const fs = require('fs').promises
const { read } = require('..')

const MAX_VOLUME = 200000
const MAX_COUNT = 50

async function main () {
  const results = []
  let checked = 0

  console.log('Reading schematics...')
  for await (const s of read()) {
    try {
      const parsed = nbt.simplify(await parseNbt(s.schematicData))
      const { Width: W, Height: H, Length: L } = parsed
      if (!W || !H || !L) continue

      const volume = W * H * L
      if (volume > MAX_VOLUME) {
        checked++
        continue
      }

      results.push({ schematicData: s.schematicData, url: s.url, title: s.title, tags: s.tags })
      console.log(`[${results.length}] "${s.title}" — ${W}×${H}×${L} (vol: ${volume})`)
    } catch (e) {
      // skip unparseable
    }

    if (results.length >= MAX_COUNT) break
    checked++
    if (checked > 500) break // don't scan the whole file
  }

  const outPath = __dirname + '/viewer/public/all_small.json'
  await fs.writeFile(outPath, JSON.stringify(results))
  console.log(`\nWrote ${results.length} schematics to viewer/public/all_small.json`)
}

main()
