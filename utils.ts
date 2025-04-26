export const cosine = (a: number[], b: number[]) => {
  let dot = 0

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
  }

  const ma = a.reduce((sum, val) => sum + Math.pow(val, 2), 0)
  const mb = b.reduce((sum, val) => sum + Math.pow(val, 2), 0)

  return dot / (Math.sqrt(ma) * Math.sqrt(mb))
}
