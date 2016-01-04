import struct
import pymongo
import pickle
from os import listdir
from os.path import isfile, join

class NyseOpenBook(object):
    format_characteristics = '>iHi11s2hih2ci2B3ih4c3i'
    symbols_dict = {}

    def __init__(self, name='unknown'):
        self.name = name

    def parse_from_binary(self, binary_record):
        format_size = struct.calcsize(self.format_characteristics)
        assert (len(binary_record) == format_size)
        data = struct.unpack(self.format_characteristics, binary_record)
        return NyseOpenBookRecord(data)

    def add_record(self, record):
        if record.Volume > 0:
            self.symbols_dict.setdefault(record.Symbol,[]).append(record)

    def read_from_file(self, file_path, record_filter=(lambda x: True), max_rows=1000):
        with open(file_path, 'rb') as file:
            binary_record = file.read(69)
            i = 0
            while (len(binary_record) > 0) & ((i < max_rows) | (max_rows == 0)):
                # parse OpenBook NYSE record

                record = self.parse_from_binary(binary_record)
                if record_filter(record):
                    self.add_record(record)

                binary_record = file.read(69)
                i += 1

                if i%100000 == 0:
                    print('items processed: ', i)

    def read_from_db(self, db, filter):
        results = db[self.name].find(filter)
        for result in results:
            record = NyseOpenBookRecord.from_db_result(result)
            self.add_record(record)

    def save_to_db(self, db):
        i = 0;
        for list in self.symbols_dict.itervalues():
            for record in list:
                item = {
                    'symbol': record.Symbol,
                    'time': record.SourceTime,
                    'volume': record.Volume,
                    'price': record.Price,
                    'ChgQty': record.ChgQty,
                    'Side': record.Side
                }
    
                if i % 100000 == 0:
                    print('processed {} items'.format(i))
    
                db[self.name].save(item)
                i += 1

    def print_records(self):
        print('|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|'.format('SYM', 'TIME', 'VOLUME', 'PRICE', 'SIDE'))

        gap = ''
        for i in range(5):
            gap += '|{0:{fill}{align}10}'.format('', fill='-', align='^')
        gap += '|'
        print(gap)

        for list in self.symbols_dict.itervalues():
            for record in list:
                print(record)

    def getXY(self):
        X = []
        y = []
        for list in self.symbols_dict.itervalues():
            for record in list:
                X.append(record.getX())
                y.append(record.getY())
        return X, y

    def pickle_to_file(self, filename):
        output = open('symbols//' + filename, 'wb')
        pickle.dump(self.symbols_dict[filename], output)
        output.close()
        
    def pickle_from_file(self, filename):
        self.name = filename
        input = open('symbols//' + filename, 'rb')
        list = pickle.load(input)
        self.symbols_dict.setdefault(filename, list)
        input.close()


class NyseOpenBookRecord(object):
    def __init__(self, data=None):
        if data:
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
        result += '|{:^10}'.format(self.SourceTime)
        result += '|{:^10}'.format(self.Volume)
        result += '|{:^10}'.format(self.Price)
        result += '|{:^10}'.format(self.Side)
        result += '|'
        return result

    def getX(self):
        return [self.Volume, self.Price, self.SourceTime, 0 if self.Side == 'S' else 1]

    def getY(self):
        return self.Price

    @classmethod
    def from_db_result(cls, result):
        empty_record = cls()
        empty_record.Symbol = result['symbol']
        empty_record.SourceTime = result['time']
        empty_record.Price = result['price']
        empty_record.Volume = result['volume']
        empty_record.ChgQty = result['ChgQty']
        empty_record.Side = result['Side']
        return empty_record


def getTestData():
    book = NyseOpenBook("test")
    db_client = pymongo.MongoClient('localhost', 27017)
    book.read_from_db(db_client['nyse'], {'symbol': 'PIP'})
    return book


def main():
    book = NyseOpenBook("test")
    # filename = 'bigFile.binary'
    filename = 'openbookultraAZ_A20130403_1_of_1'
    # record_filter = (lambda x: ('NOM' in x.Symbol) & ((x.Side == 'B') | (x.Side == 'S')))
    # record_filter = (lambda x: 'PIP' in x.Symbol)
    record_filter = (lambda x: True)
    book.read_from_file(filename, record_filter, 5000)
    # book.print_records()
    db_client = pymongo.MongoClient('localhost', 27017)
    book.save_to_db(db_client['nyse'])
      
    for list in book.symbols_dict.itervalues():
        book.pickle_to_file(list[0].Symbol)
     
    symbols = [f for f in listdir("symbols") if isfile(join('symbols', f))]
    
    book.symbols_dict = {}
    book.pickle_from_file('PIP')
    # db.test.aggregate({$group: {_id : "$symbol", count: {$sum : 1}}}, { $sort: {count: -1} });
    # book.read_from_db(db_client['nyse'], {'symbol': 'PIP'})
    book.print_records()

if __name__ == '__main__':
    main()