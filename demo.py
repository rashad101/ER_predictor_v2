value1 = 'one'
value2 = 'two'
d = {
        'key1': 1,
        'key2': 2,
        'key3': 3
    }
CSV ="\n".join([str(v)+','+k for k,v in d.items()])
with open("filename.csv", "w") as file:
    file.write(CSV)
