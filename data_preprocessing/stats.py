__author__ = 'yuhongliang324'
import os
import xlrd
from annotation import excel_root


data_root = '/Users/yuhongliang324/Documents/Lab/Dataset/Rapport/'


def count_success2(root):
    files = os.listdir(root)
    files.sort()
    left_suc, right_suc = 0, 0
    left_total, right_total = 0, 0
    for fn in files:
        if not fn.endswith('txt'):
            continue
        fn_path = root + fn
        reader = open(fn_path)
        lines = reader.readlines()
        reader.close()
        num_suc, total = 0, 0
        lines = map(lambda x: x.strip(), lines)
        lines = lines[1:]
        for line in lines:
            if len(line) < 10:
                continue
            suc = line.split(',  ')[3]
            if suc == '1':
                num_suc += 1
            total += 1

        success_rate = float(num_suc) / float(total)
        if 'left' in fn:
            left_total += 1
            left_suc += success_rate
        else:
            right_total += 1
            right_suc += success_rate
    left_suc /= left_total
    right_suc /= right_total
    return left_suc, left_total, right_suc, right_total


def count_success():
    files = os.listdir(data_root)
    files.sort()
    left_suc, left_total, right_suc, right_total = 0, 0, 0, 0
    for fn in files:
        if not os.path.isdir(data_root + fn):
            continue
        ls, lt, rs, rt = count_success2(data_root + '/' + fn)
        left_suc += ls * lt
        right_suc += rs * rt
        left_total += lt
        right_suc += rt
    left_suc /= left_total
    right_suc /= right_total
    print 'left', left_suc, left_total
    print 'right', right_suc, right_total


def rating_distribution(excel_root):
    rating_files = os.listdir(excel_root)
    rating_files.sort()
    offset = 1
    numr = [0 for i in xrange(8)]
    total = 0
    for rating_file in rating_files:
        if not (rating_file.startswith('D') and rating_file.endswith('xlsx')):
            continue
        file_path = os.path.join(excel_root, rating_file)
        print file_path
        workbook = xlrd.open_workbook(file_path)
        worksheet = workbook.sheet_by_index(0)

        for i, row in enumerate(range(worksheet.nrows)):
            if i < offset:
                continue
            if worksheet.cell_type(i, 0) in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK) or \
                worksheet.cell_type(i, 1) in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK):
                continue
            rating = int(worksheet.cell(i, 1).value)
            numr[rating] += 1
            total += 1

    print map(lambda x: float(x) / total, numr), total
    print numr


def test1():
    count_success()


def test2():
    rating_distribution(excel_root)

if __name__ == '__main__':
    test2()
