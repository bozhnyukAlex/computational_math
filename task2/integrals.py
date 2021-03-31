from math import pi, cos, log

def integralLeft(a, b, n, f):
    h = (b - a) / n
    x = [a + k * h for k in range(0, n + 1)]
    result = 0
    for i in range(0, n):
        result += (x[i + 1] - x[i]) * f(x[i])
    return result

def integralRight(a, b, n, f):
    h = (b - a) / n
    x = [a + k * h for k in range(0, n + 1)]
    result = 0
    for i in range(0, n):
        result += (x[i + 1] - x[i]) * f(x[i + 1])
    return result


def integralMiddle(a, b, n, f):
    h = (b - a) / n
    x = [a + k * h for k in range(0, n + 1)]
    result = 0
    for i in range(0, n):
        result += (x[i + 1] - x[i]) * (f(x[i + 1]) + f(x[i]) / 2)
    return result

f1 = lambda x: log(1 + 2 * 0.1 * cos(x) + 0.1 ** 2)
f2 = lambda x: log(1 + 2 * 0.9 * cos(x) + 0.9 ** 2)
n = 10000
il1 = integralLeft(0, pi, n, f1)
il2 = integralLeft(0, pi, n, f2)
im1 = integralMiddle(0, pi, n, f1)
im2 = integralMiddle(0, pi, n, f2)
ir1 = integralRight(0, pi, n, f1)
ir2 = integralRight(0, pi, n, f2)
print(il1)
print(il2)
print(im1)
print(im2)
print(ir1)
print(ir2)

## Quad

def momentFun(i, p):
    return lambda x: p(x) * (x ** i)


def quad(p, f, a, b):
    n = 100000
    c = [integralMiddle(a, b, n, momentFun(i, p)) for i in range(4)]
    K1 = np.array([[c[1], c[0]], [c[2], c[1]]])
    b1 = np.array([-c[2], -c[3]])
    sr = np.linalg.solve(K1, b1)
    s = sr[0]
    r = sr[1]
    x = np.roots([1, s, r])
    K2 = np.array([[1, 1], [x[0], x[1]]])
    b2 = np.array([c[0], c[1]])
    A = np.linalg.solve(K2, b2)
    return A[0] * f(x[0]) + A[1] * f(x[1])
    
pa1 = lambda x: x ** 0.5
pa2 = lambda x: x ** 2
pa3 = lambda x: x ** 3
pb = lambda x: exp(x)
pc = lambda x: log(x)
f = lambda _: 1

qa1 = quad(pa1, f, 0, 1)
qa2 = quad(pa2, f, 0, 1)
qa3 = quad(pa3, f, 0, 1)
qb = quad(pb, f, 0, 0.5)
qc = quad(pc, f, 0, 0.5)

print('Моя версия: ')
print(qa1)
print(qa2)
print(qa3)
print(qb)
print(qc)


# Check

fpa1 = lambda x: (x ** 0.5) * f(x)
fpa1 = lambda x: (x ** 0.5) * f(x)
fpa2 = lambda x: (x ** 2) * f(x)
fpa3 = lambda x: (x ** 3) * f(x)
fpb = lambda x: exp(x) * f(x)
fpc = lambda x: log(x) * f(x)

sqa1 = (integrate.quad(fpa1, 0, 1))[0]
sqa2 = (integrate.quad(fpa2, 0, 1))[0]
sqa3 = (integrate.quad(fpa3, 0, 1))[0]
sqb = (integrate.quad(fpb, 0, 0.5))[0]
sqc = (integrate.quad(fpc, 0, 0.5))[0]

print("Версия с scipy:")
print(sqa1)
print(sqa2)
print(sqa3)
print(sqb)
print(sqc)

dqa1 = abs(sqa1 - qa1)
dqa2 = abs(sqa2 - qa2)
dqa3 = abs(sqa3 - qa3)
dqb = abs(sqb - qb)
dqc = abs(sqc - qc)
print("")

print("Проверка правильности - разности интегралов: ")
print(dqa1)
print(dqa2)
print(dqa3)
print(dqb)
print(dqc)

### Применение 
# наша функция
F = lambda x: exp(x)
# весовая функция
p = lambda x: x ** 2 
# левая граница
A = 0
# правая граница
B = 1
# Используем функцию quad, которая как раз 
q = quad(p, F, A, B)
print("Результат: ", q[0])
print("Коэффициенты А:", q[1])
print("Узлы x: ", q[2])