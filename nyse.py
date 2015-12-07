import struct


class NyseOpenBook(object):
    format_characteristics = '>iHi11s2hih2ci2B3ih4c3i'
    records = []

    def parse_from_binary(self, binary_record):
        format_size = struct.calcsize(self.format_characteristics)
        assert (len(binary_record) == format_size)
        data = struct.unpack(self.format_characteristics, binary_record)
        return NyseOpenBookRecord(data)

    def add_record(self, record):
        self.records.append(record)

    def read_from_file(self, file_path, record_filter=(lambda x: True), max_rows=1000):
        with open(file_path, 'rb') as file:
            binary_record = file.read(69)
            i = 0
            while (len(binary_record) > 0) & (i < max_rows):
                # parse OpenBook NYSE record

                record = self.parse_from_binary(binary_record)
                if record_filter(record):
                    self.add_record(record)

                binary_record = file.read(69)
                i += 1

    def print_records(self):
        print('|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|'.format('SYM', 'TIME', 'VOLUME', 'PRICE', 'SIDE'))

        gap = ''
        for i in range(5):
            gap += '|{0:{fill}{align}10}'.format('', fill='-', align='^')
        gap += '|'
        print(gap)

        for rec in self.records:
            print(rec)


class NyseOpenBookRecord(object):
    def __init__(self, data):
        self.MsgSeqNum = data[0]
        self.MsgType = data[1]
        self.SendTime = data[2]
        self.Symbol = str(data[3].partition(b'\0')[0].decode('utf8'))
        self.MsgSize = data[4]
        self.SecurityIndex = data[5]
        self.SourceTime = data[6] * 1000 + data[7]
        self.QuoteCondition = data[8]
        self.TradingStatus = data[9]
        self.SourceSeqNum = data[10]
        self.SourceSessionID = data[11]
        self.Price = float(data[13]) / (10.0 ** data[12])
        self.Volume = data[14]
        self.ChgQty = data[15]
        self.Side = str(data[17].decode('utf8'))
        self.ReasonCode = data[19]

    def __str__(self):
        result = ''
        result += '|{:^10}'.format(self.Symbol)
        result += '|{:^10}'.format(self.SendTime)
        result += '|{:^10}'.format(self.Volume)
        result += '|{:^10}'.format(self.Price)
        result += '|{:^10}'.format(self.Side)
        result += '|'
        return result


def main():
    book = NyseOpenBook()
    filename = 'bigFile.binary'
    # record_filter = (lambda x: ('NOM' in x.Symbol) & ((x.Side == 'B') | (x.Side == 'S')))
    # record_filter = (lambda x: 'CUR' in x.Symbol)
    record_filter = (lambda x: True)
    book.read_from_file(filename, record_filter, 10)
    book.print_records()


if __name__ == '__main__':
    main()