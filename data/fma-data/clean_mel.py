import csv

with open('fma_metadata/melspectrogram.csv', 'r') as f_raw:
    with open('fma_metadata/melspectrogram2.csv', 'w') as f:
        lines_raw = list(csv.reader(f_raw))

        line_count = len(lines_raw)
        if len(lines_raw[-1]) == 0:
            line_count -= 1

        for i in range(line_count):
            if lines_raw[i][1] == '':
                if i+1 < line_count and lines_raw[i+1][1] != '':
                    f.write(lines_raw[i][0])
                    for data in lines_raw[i][1:]:
                        f.write(',{}'.format(data))
                    f.write('\n')
                else:
                    print('Skipping line: {}'.format(lines_raw[i][0:5]))
            else:
                f.write(lines_raw[i][0])
                for data in lines_raw[i][1:]:
                    f.write(',{}'.format(data))
                f.write('\n')
