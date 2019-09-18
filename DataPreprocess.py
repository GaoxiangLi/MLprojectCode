# use the csv module
import csv
import numpy as np


def dataprocess(args):
    ds = []
    filename = args.filename
    with open("%s" % (filename)) as f:
        print(1)
        reader = csv.reader(f)
        for row in reader:
            ds.append(row)

    # show the data
    header = args.header
    index = args.index
    label = args.label
    index_1 = args.Consequence_index
    index_2 = args.Segway_index
    index_3 = args.Label_index
    x_begin = 0
    y_begin = 0
    if header:
        attributes = ds[0]
        x_begin = 1
        index_1, index_2 = attributes.index('Consequence'), attributes.index('Segway')
        if index:
            y_begin = 1
        if label:
            index_3 = attributes.index('Newlabel')
    f_res = []
    l_res = []
    for i in range(x_begin, len(ds)):
        tem1 = []
        tem2 = []
        for j in range(y_begin, len(ds[x_begin])):
            if j == index_1 or j == index_2 or j == index_3:
                if j == index_1:
                    if ds[i][j] == '3PRIME_UTR':
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                    if ds[i][j] == '5PRIME_UTR':
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(1)
                    if ds[i][j] == 'CANONICAL_SPLICE':
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(1)
                        tem1.append(0)
                    if ds[i][j] == 'DOWNSTREAM':
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(1)
                        tem1.append(1)
                    if ds[i][j] == 'INTERGENIC':
                        tem1.append(0)
                        tem1.append(1)
                        tem1.append(0)
                        tem1.append(0)
                    if ds[i][j] == 'INTRONIC':
                        tem1.append(0)
                        tem1.append(1)
                        tem1.append(0)
                        tem1.append(1)
                    if ds[i][j] == 'REGULATORY':
                        tem1.append(0)
                        tem1.append(1)
                        tem1.append(1)
                        tem1.append(0)
                    if ds[i][j] == 'SPLICE_SITE':
                        tem1.append(0)
                        tem1.append(1)
                        tem1.append(1)
                        tem1.append(1)
                    if ds[i][j] == 'UPSTREAM':
                        tem1.append(1)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                if j == index_2:
                    if ds[i][j] == 'c0':
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                    if ds[i][j] == 'c0':
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                    if ds[i][j] == 'c0':
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                    if ds[i][j] == 'c0':
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                    if ds[i][j] == 'c0':
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                    if ds[i][j] == 'c0':
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                    if ds[i][j] == 'c0':
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                    if ds[i][j] == 'c0':
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                    if ds[i][j] == 'c0':
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                    if ds[i][j] == 'c0':
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                    if ds[i][j] == 'c0':
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                    if ds[i][j] == 'c0':
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                    if ds[i][j] == 'c0':
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                    if ds[i][j] == 'c0':
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                        tem1.append(0)
                if j == index_3:
                    if ds[i][j] == 0:
                        tem2.append(1)
                        tem2.append(0)
                    else:
                        tem2.append(0)
                        tem2.append(1)

            else:
                tem1.append(ds[i][j])

        f_res.append(tem1)
        l_res.append(tem2)
    f_res = np.array(f_res)
    l_res = np.array(l_res)
    f_res = np.reshape(f_res, [-1, 74])
    l_res = np.reshape(l_res, [-1, 2])
    # print(f_res.shape)
    f_name = args.feature_file
    l_name = args.label_file
    print(f_res)
    print(l_res)
    np.savetxt("%s" % (f_name), f_res, delimiter=",")
    np.savetxt("%s" % (l_name), l_res, delimiter=",")


def main(argv=None):
    dataprocess()


if __name__ == '__main__':
    main()
