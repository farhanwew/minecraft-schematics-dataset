const nbt = require('prismarine-nbt')
const promisify = require('util').promisify
const parseNbt = promisify(nbt.parse)
const { read } = require('.')

// Map block IDs to ASCII chars for visual distinction
const BLOCK_CHARS = {
  0: '·',   // air
  1: '#',   // stone
  2: 'G',   // grass
  3: 'd',   // dirt
  4: '#',   // cobblestone
  5: 'W',   // wood planks
  8: '~',   // water
  9: '~',   // still water
  10: '^',  // lava
  11: '^',  // still lava
  12: 's',  // sand
  13: 'g',  // gravel
  17: 'T',  // log
  18: 'L',  // leaves
  20: 'o',  // glass
  35: 'w',  // wool
  54: 'C',  // chest
  80: '*',  // snow
  89: 'l',  // glowstone
  95: 'O',  // stained glass
}

function blockChar (id) {
  const b = id < 0 ? id + 256 : id
  return BLOCK_CHARS[b] || (b === 0 ? '·' : '█')
}

function printLayer (blocks, width, length, y) {
  const rows = []
  for (let z = 0; z < length; z++) {
    let row = ''
    for (let x = 0; x < width; x++) {
      const idx = y * width * length + z * width + x
      row += blockChar(blocks[idx])
    }
    rows.push(row)
  }
  return rows.join('\n')
}

function findInterestingLayer (blocks, width, length, height) {
  let bestY = 0
  let bestNonAir = 0
  for (let y = 0; y < height; y++) {
    let nonAir = 0
    for (let z = 0; z < length; z++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width * length + z * width + x
        const b = blocks[idx] < 0 ? blocks[idx] + 256 : blocks[idx]
        if (b !== 0) nonAir++
      }
    }
    if (nonAir > bestNonAir) {
      bestNonAir = nonAir
      bestY = y
    }
  }
  return { y: bestY, nonAir: bestNonAir }
}

async function main () {
  const LIMIT = 5          // how many schematics to preview
  const MAX_DIM = 60       // cap width/length for display

  let i = 0
  for await (const s of read()) {
    const parsed = nbt.simplify(await parseNbt(s.schematicData))
    const { Width: W, Height: H, Length: L, Blocks } = parsed

    if (!Blocks || !W || !H || !L) {
      console.log(`[${i + 1}] "${s.title}" — no block data, skipping\n`)
      i++
      if (i >= LIMIT) break
      continue
    }

    const { y, nonAir } = findInterestingLayer(Blocks, W, L, H)

    const displayW = Math.min(W, MAX_DIM)
    const displayL = Math.min(L, MAX_DIM)

    console.log('═'.repeat(displayW + 2))
    console.log(`[${i + 1}] "${s.title}"`)
    console.log(`    Size: ${W}W × ${H}H × ${L}L  |  Tags: ${(s.tags || []).join(', ') || '—'}`)
    console.log(`    User: ${s.user}  |  Diamonds: ${s.diamondCount}  |  Downloads: ${s.downloads}`)
    console.log(`    Showing layer Y=${y} (most filled, ${nonAir} non-air blocks)`)
    if (W > MAX_DIM || L > MAX_DIM) console.log(`    (cropped to ${displayW}×${displayL})`)
    console.log('─'.repeat(displayW + 2))

    // Print the layer, crop large schematics
    for (let z = 0; z < displayL; z++) {
      let row = '│'
      for (let x = 0; x < displayW; x++) {
        const idx = y * W * L + z * W + x
        row += blockChar(Blocks[idx])
      }
      row += '│'
      console.log(row)
    }

    console.log('═'.repeat(displayW + 2))
    console.log()

    i++
    if (i >= LIMIT) break
  }
}

main()
