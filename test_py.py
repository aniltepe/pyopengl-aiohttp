# from objects import obj_male

WIDTH, HEIGHT = 1024, 1024
lips_conf_file = open("np_lipsconf.txt", "r")
lips_conf = lips_conf_file.readlines()
lips_tri_indexes = [int(line.split(" ")[0]) for line in lips_conf]
s = ""
for i in range(9884):
    if i in lips_tri_indexes:
        idx = lips_tri_indexes.index(i)
        values_str = lips_conf[idx]
        values_int = [int(s) for s in values_str.split(" ")]
        values_int.pop(0)
        values_uv = [values_int[index]/WIDTH if index%2 == 0 else (HEIGHT-values_int[index])/HEIGHT for index in range(len(values_int))]
        s += f"{values_uv[0]}, {values_uv[1]}, {values_uv[2]}, {values_uv[3]}, {values_uv[4]}, {values_uv[5]},\n"
    else:
        s += "0.0, 0.0, 0.0, 0.0, 0.0, 0.0,\n"

f = open("tex_male.txt", "w")
f.write(s)
f.close()
# print(len(obj_male.indice))