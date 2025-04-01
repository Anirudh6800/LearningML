
m1 = [[1,2,3],[4,5,6]]
m2 = [[1,2],[3,4],[5,6]]

def matmul(m1,m2):

  p = len(m1)
  q1 = len(m1[0])
  q2 = len(m2)
  r = len(m2[0])

  if q1 == q2:
    q = q1

    m3row = []
    m3 = []

    for i in range(p):
      for j in range(r):
        m3row.append(0)
      m3.append(m3row)
      m3row = []

    print(m3)

    for k in range(p):
      for j in range(r):
        for i in range(q):
          m3[k][j] = m3[k][j] + m1[k][i] * m2[i][j]
  else:
    print("Dimension mismatch")
    return

  return m3

result = matmul(m1,m2)
print(result)
