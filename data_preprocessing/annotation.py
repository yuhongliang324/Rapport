__author__ = 'yuhongliang324'
import os
import xlrd

root = '/Users/yuhongliang324/Dropbox/RAPT/'

excel_root = root + 'labels/'
label_file = root + 'labels.txt'


def customize_labels(excel_root, outfile):
    rating_files = os.listdir(excel_root)
    rating_files.sort()
    offset = 1
    fw = open(outfile, 'wb')
    fw.write('Slice\tRating\tConfidence\n')
    for rating_file in rating_files:
        if not (rating_file.startswith('D') and rating_file.endswith('xlsx')):
            continue
        file_path = os.path.join(excel_root, rating_file)
        print file_path
        workbook = xlrd.open_workbook(file_path)
        worksheet = workbook.sheet_by_index(0)

        splits = rating_file.split('_')
        rating_name = splits[0]+'_'+splits[1]
        ID_line = {}
        maxID = 0

        for i, row in enumerate(range(worksheet.nrows)):
            if i < offset:
                continue
            if worksheet.cell_type(i, 0) in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK) or \
                worksheet.cell_type(i, 1) in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK):
                continue
            sliceID, rating, conf = int(worksheet.cell(i, 0).value),\
                                    int(worksheet.cell(i, 1).value), int(worksheet.cell(i, 2).value)
            if sliceID > maxID:
                maxID = sliceID
            line = rating_name + '_' + str(sliceID).zfill(3) + '\t' + str(rating) + '\t' + str(conf) + '\n'
            ID_line[sliceID] = line
        for ID in xrange(maxID + 1):
            if ID in ID_line:
                fw.write(ID_line[ID])
        print rating_name, maxID
    fw.close()


def write_success2(root, outdir):
    files = os.listdir(root)
    files.sort()
    left_suc, right_suc = 0, 0
    left_total, right_total = 0, 0
    for fn in files:
        if not fn.endswith('txt'):
            continue
        tmpfn = fn
        if tmpfn.startswith('D'):
            tmpfn = '1_' + tmpfn
        sp = tmpfn.split('_')
        outfn = sp[1] + '_' + sp[2]
        sliceID = sp[4].split('.')[0]
        if sliceID.startswith('Slice'):
            sliceID = sliceID[5:]
        outfn += '_' + sliceID.zfill(3)
        if sp[3] == 'left':
            outfn = 'L_' + outfn
        else:
            outfn = 'R_' + outfn
        outfn += '.txt'
        print outfn
        writer = open(os.path.join(outdir, outfn), 'w')

        fn_path = os.path.join(root, fn)
        reader = open(fn_path)
        lines = reader.readlines()
        reader.close()
        num_suc, total = 0, 0
        lines = map(lambda x: x.strip(), lines)
        lines = lines[1:]
        for line in lines:
            if len(line) < 10:
                continue
            sp = line.split(',  ')
            frameID, suc = sp[0], sp[3]
            if suc == '1':
                num_suc += 1
            writer.write(frameID + '\t' + suc + '\n')
            total += 1

        writer.close()
        success_rate = float(num_suc) / float(total)
        if 'left' in fn:
            left_total += 1
            left_suc += success_rate
        else:
            right_total += 1
            right_suc += success_rate
    left_suc /= left_total
    right_suc /= right_total
    print root, left_suc, left_total, right_suc, right_total


def write_success(data_root, success_dir):
    files = os.listdir(data_root)
    files.sort()
    for fn in files:
        if not os.path.isdir(os.path.join(data_root, fn)):
            continue
        write_success2(os.path.join(data_root, fn), success_dir)


def test1():
    customize_labels(excel_root, label_file)


def test2():
    both_ori_features_dir = '/usr0/home/liangkeg/RAPT_dataset/features/ori_features/'
    success_dir = '/usr0/home/liangkeg/RAPT_dataset/features/success'
    write_success(both_ori_features_dir, success_dir)

if __name__ == '__main__':
    test2()
