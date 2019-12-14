
def preprocess_csv(csv_filepath, csv_filepath_out, whichone='training'):
    csv_file_in = open(csv_filepath, 'r')
    csv_file_out = open(csv_filepath_out, 'w')

    if whichone == 'training':
        for each_line in csv_file_in.readlines():
            line = each_line.split(',')
            # as crim, indus and tax (and medv, of course) are the only label columns we are interested in

            line_out = line[0] + ',' + line[2] + ',' + line[9] + ',' + line[13]  # line[13] contains a'\n'
            csv_file_out.write(line_out)

    elif whichone == 'test':
        for each_line in csv_file_in.readlines():
            line = each_line.split(',')
            # as crim, indus and tax (and medv, of course) are the only label columns we are interested in

            line_out = line[0] + ',' + line[2] + ',' + line[7] + ',' + line[9]  # line[9] contains a'\n'
            csv_file_out.write(line_out)

    elif whichone == 'predict':
        for each_line in csv_file_in.readlines():
            line = each_line.split(',')
            # as crim, indus and tax (and medv, of course) are the only label columns we are interested in

            line_out = line[0] + ',' + line[2] + ',' + line[7]  # line[7] doesn't contain a'\n' here
            csv_file_out.write(line_out + '\n')

CSV_FILE = './BostonHousing.csv'
CSV_FILE_OUT = './BostonHousing_subset.csv'

preprocess_csv('./BostonHousing.csv', './BostonHousing_subset.csv', 'training')
preprocess_csv('./boston_test.csv', './boston_test_subset.csv', 'test')
preprocess_csv('./boston_predict.csv', './boston_predict_subset.csv', 'predict')