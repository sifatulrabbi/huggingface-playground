import json

with open("./json_output.txt", "r") as f:
    data = []
    line = f.readline()
    currobj = {}

    while line:
        if line.startswith("-"):
            data.append(currobj)
            currobj = {}
        elif line.startswith("{"):
            currobj["input"] = line.strip()
        else:
            currobj["output"] = line.strip()
        line = f.readline()

    with open("./json_output.json", "w") as outfile:
        json.dump(data, outfile)
    print("Total examples:", len(data))
