from objects import obj_lefteye

s = ""
obj = obj_lefteye
for i, p in enumerate(obj.position):
    if i % 3 == 1:
        p += -0.001
    if i % 3 == 2:
        p += 0.002
    s += f"{p}, "

f = open("np_lefteye.txt", "w")
f.write(s)
f.close()
# print(len(obj_male.indice))