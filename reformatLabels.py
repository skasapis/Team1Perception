outfile = open("Team1_fixed10.txt", "w")
outfile.write('guid/image,label\n')

with open("Team1.txt") as fin:
    data = fin.readlines()
    for line in data:
        words = line.split(',')
        print words
        outfile.write(words[0])
        outfile.write(',')
        outfile.write(str(int(words[1])-1))
        outfile.write('\n')

fin.close()
outfile.close()


