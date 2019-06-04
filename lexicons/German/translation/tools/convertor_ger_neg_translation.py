
words = []
with open("test.txt") as f:
    for line in f:
        # Split over |
        first_split = line.split("|")
        # print(first_split[0])
        first_word = first_split[0].split(" ")[0]
        # Split right side over blank char
        second_split = first_split[1].split(" ")
        number = second_split[2]
        # For new line cutting
        number = (number.split("\n"))[0]
        number = number.replace(",", ".")
        number = float(number)
        words.append({"word": first_word, "weight": number})
        cnt = 3
        while cnt < len(second_split):
            word = second_split[cnt].split(",")[0]
            word = word.split("\n")[0]
            words.append({"word": word, "weight": number})
            cnt += 1

words.sort(key=lambda x: x["word"])

prev = 0
cnt = 1
prev_vals = []
while cnt < len(words):
    if words[prev]["word"] == words[cnt]["word"]:
        prev_vals.append(words[cnt]["weight"])
        words.remove(words[cnt])
    else:
        if len(prev_vals) > 0:
            summ = 0
            for i in prev_vals:
                summ += i
            summ += words[prev]["weight"]
            avg_weight = summ / (len(prev_vals) + 1)
            words[prev]["weight"] = avg_weight
            prev_vals = []
        cnt += 1
        prev += 1
if len(prev_vals) > 0:
    summ = 0
    for i in prev_vals:
        summ += i
    summ += words[prev]["weight"]
    avg_weight = summ / (len(prev_vals) + 1)
    words[prev]["weight"] = avg_weight
    prev_vals = []

with open("test_out.txt", "w") as f:
    cnt = 0
    while cnt < len(words):
        cnt += 1
        if cnt == len(words):
            f.write("%s,%.4f" % (words[cnt-1]["word"], words[cnt-1]["weight"]))
        else:
            f.write("%s,%.4f\n" % (words[cnt-1]["word"], words[cnt-1]["weight"]))
