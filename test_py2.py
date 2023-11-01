from objects import obj_righteye

s = ""
obj = obj_righteye
for i in range(0, len(obj.indice), 3):
    s += f"{obj.texture[obj.indice[i] * 2] if obj.texture[obj.indice[i] * 2] > 0.0 else 0.0}, "
    s += f"{obj.texture[obj.indice[i] * 2 + 1] if obj.texture[obj.indice[i] * 2 + 1] > 0.0 else 0.0},\n"
    s += f"{obj.texture[obj.indice[i + 1] * 2] if obj.texture[obj.indice[i + 1] * 2] > 0.0 else 0.01}, "
    s += f"{obj.texture[obj.indice[i + 1] * 2 + 1] if obj.texture[obj.indice[i + 1] * 2 + 1] > 0.0 else 0.0},\n"
    s += f"{obj.texture[obj.indice[i + 2] * 2] if obj.texture[obj.indice[i + 2] * 2] > 0.0 else 0.0}, "
    s += f"{obj.texture[obj.indice[i + 2] * 2 + 1] if obj.texture[obj.indice[i + 2] * 2 + 1] > 0.0 else 0.01},\n"

f = open("tex_righteye.txt", "w")
f.write(s)
f.close()
# print(len(obj_male.indice))