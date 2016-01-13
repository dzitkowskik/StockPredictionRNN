import struct
import pymongo
import pickle
import numpy as np
from keras.utils import np_utils

class Data:
    def __init__(self, x, y):
        self.x = x
        self.y = y

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

    def read_from_file(self, file_path, record_filter=(lambda x: True), max_rows=10000):
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
#             self.MsgSeqNum = data[0]
#             self.MsgType = data[1]
#             self.SendTime = data[2]
            self.Symbol = str(data[3].partition(b'\0')[0].decode('utf8'))
#             self.MsgSize = data[4]
#             self.SecurityIndex = data[5]
            self.SourceTime = data[6] * 1000 + data[7]
#             self.QuoteCondition = data[8]
#             self.TradingStatus = data[9]
#             self.SourceSeqNum = data[10]
#             self.SourceSessionID = data[11]
            self.Price = float(data[13]) / (10.0 ** data[12])
            self.Volume = data[14]
#             self.ChgQty = data[15]
            self.Side = str(data[17].decode('utf8'))
#             self.ReasonCode = data[19]

    def __str__(self):
        result = ''
        result += '|{:^10}'.format(self.Symbol)
        result += '|{:^10}'.format(self.SourceTime)
        result += '|{:^10}'.format(self.Volume)
        result += '|{:^10}'.format(self.Price)
        result += '|{:^10}'.format(self.Side)
        result += '|'
        return result

    @classmethod
    def from_db_result(cls, result):
        empty_record = cls()
        empty_record.Symbol = result['symbol']
        empty_record.SourceTime = result['time']
        empty_record.Price = result['price']
        empty_record.Volume = result['volume']
#         empty_record.ChgQty = result['ChgQty']
        empty_record.Side = result['Side']
        return empty_record


class NyseOrderBook(object):
    buy_orders = []
    sell_orders = []
    
    X = []
    Y = []
        
    transaction_price = 0.0
    prev_transaction_price = 0.0
    
    def __init__(self, name='unknown'):
        self.name = name

    def process_order(self, order):
        remaining_volume = order.Volume
        
        if order.Side == 'B':
            while remaining_volume > 0:
                if self.sell_orders:
                    if self.sell_orders[0].Price <= order.Price:
                        if self.sell_orders[0].Volume <= remaining_volume:
                            remaining_volume -= self.sell_orders.pop(0).Volume
                        else:
                            self.sell_orders[0].Volume -= remaining_volume
                            remaining_volume = 0
                        self.transaction_price = order.Price
                    else:
                        matching_order = next((x for x in self.buy_orders if x.Price == order.Price), None)
                        if matching_order:
                            matching_order.Volume += order.Volume
                        else:    
                            self.buy_orders.append(order)
                            self.buy_orders.sort(key=lambda x: x.Price, reverse=True)
                        break
                else:
                    matching_order = next((x for x in self.buy_orders if x.Price == order.Price), None)
                    if matching_order:
                        matching_order.Volume += order.Volume
                    else:    
                        self.buy_orders.append(order)
                        self.buy_orders.sort(key=lambda x: x.Price, reverse=True)
                    break
        elif order.Side == 'S':
            while remaining_volume > 0:
                if self.buy_orders:
                    if self.buy_orders[0].Price >= order.Price:
                        if self.buy_orders[0].Volume <= remaining_volume:
                            remaining_volume -= self.buy_orders.pop(0).Volume
                        else:
                            self.buy_orders[0].Volume -= remaining_volume
                            remaining_volume = 0
                        self.transaction_price = order.Price
                    else:
                        matching_order = next((x for x in self.sell_orders if x.Price == order.Price), None)
                        if matching_order:
                            matching_order.Volume += order.Volume
                        else:                   
                            self.sell_orders.append(order)
                            self.sell_orders.sort(key=lambda x: x.Price, reverse=False)
                        break
                else:
                    matching_order = next((x for x in self.sell_orders if x.Price == order.Price), None)
                    if matching_order:
                        matching_order.Volume += order.Volume
                    else:                   
                        self.sell_orders.append(order)
                        self.sell_orders.sort(key=lambda x: x.Price, reverse=False)
                    break
                
        self.update_history(order)
        
    def update_history(self, order):
        levels = 3
        n = levels
        
        if n + 1 > len(self.sell_orders):
            n = len(self.sell_orders) - 1
        
        if n + 1 > len(self.buy_orders):
            n = len(self.buy_orders) - 1
        
        v1 = []
        v2 = []
        v3 = []
        v4 = [0.0, 0.0, 0.0, 0.0]
        v5 = [0.0, 0.0]
        
        for i in range(0, n):
            v1.append(self.sell_orders[i].Price)
            v1.append(self.sell_orders[i].Volume)
            v1.append(self.buy_orders[i].Price)
            v1.append(self.buy_orders[i].Volume)
            
            v2.append(self.sell_orders[i].Price - self.buy_orders[i].Price)
            v2.append((self.sell_orders[i].Price + self.buy_orders[i].Price) / 2.0)

            v3.append(abs(self.sell_orders[i+1].Price - self.sell_orders[i].Price))
            v3.append(abs(self.buy_orders[i+1].Price - self.buy_orders[i].Price))
        
        for i in range(0, n):
            v4[0] += self.sell_orders[i].Price
            v4[1] += self.buy_orders[i].Price
            v4[2] += self.sell_orders[i].Volume
            v4[3] += self.buy_orders[i].Volume
            v5[0] += self.sell_orders[i].Price - self.buy_orders[i].Price
            v5[1] += self.sell_orders[i].Volume - self.buy_orders[i].Volume
        
        if n > 0:
            v4[0] /= float(n)
            v4[1] /= float(n)
            v4[2] /= float(n)
            v4[3] /= float(n)
            
        X = self.getX(v1, v2, v3, v4, v5)
        Y = self.getY(self.prev_transaction_price, self.transaction_price)
        if len(X) == 4*levels + 2*levels + 2*levels + 4 + 2:
            self.X.append(X)
            self.Y.append(Y)
        
        self.prev_transaction_price = self.transaction_price
        
    def getX(self, v1, v2, v3, v4, v5):
        x = []
        x.extend(v1)
        x.extend(v2)
        x.extend(v3)
        x.extend(v4)
        x.extend(v5)
        return x

    def getY(self, previous, current):
        y = 1
        if current < previous:
            y = 0
        elif current > previous:
            y = 2
        return y
    
    def getXY(self):
        return self.X, self.Y


def get_balanced_subsample(x, y, subsample_size=1.0):
        class_xs = []
        min_elems = None
    
        for yi in np.unique(y):
            elems = x[(y == yi)]
            class_xs.append((yi, elems))
            if min_elems == None or elems.shape[0] < min_elems:
                min_elems = elems.shape[0]
    
        use_elems = min_elems
        if subsample_size < 1:
            use_elems = int(min_elems*subsample_size)
    
        xs = []
        ys = []
    
        for ci,this_xs in class_xs:
            if len(this_xs) > use_elems:
                np.random.shuffle(this_xs)
    
            x_ = this_xs[:use_elems]
            y_ = np.empty(use_elems)
            y_.fill(ci)
    
            xs.append(x_)
            ys.append(y_)
    
        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
    
        return xs, ys


def prepare_data(book, window_size):
    x, y = book.getXY()
    x_temp = []
    y_temp = []
    for i in range(len(x)-window_size):
        x_temp.append(x[i:(i+window_size)])
        y_temp.append(y[i+window_size])

    x = np.array(x_temp)
    y = y_temp
    
    x, y = get_balanced_subsample(x, y)
    
    xy = list(zip(x, y))
    np.random.shuffle(xy)
    x_, y_ = zip(*xy)
    x = np.array(x_)
    
    y = np_utils.to_categorical(y_, 3)

    print("{0} records with price down".format(sum(y[:, 0])))
    print("{0} records with price stable".format(sum(y[:, 1])))
    print("{0} records with price down".format(sum(y[:, 2])))
    print('x shape:', x.shape)
    print('y shape:', y.shape)

    return Data(x, y)


def get_test_data(window_size):
    book = NyseOpenBook("test")
    book.pickle_from_file('AIG')
    
    order_book = NyseOrderBook("AIG")
    for list in book.symbols_dict.itervalues():
        for order in list:
            order_book.process_order(order)
            
    return prepare_data(order_book, window_size)


def main():
    book = NyseOpenBook("test")
#     # filename = 'bigFile.binary'
#     filename = 'openbookultraAA_N20130403_1_of_1'
#     # record_filter = (lambda x: ('NOM' in x.Symbol) & ((x.Side == 'B') | (x.Side == 'S')))
#     # record_filter = (lambda x: 'AZN' in x.Symbol)
#     record_filter = (lambda x: True)
#     # record_filter = (lambda x: 'C' in x.ReasonCode)
#     book.read_from_file(filename, record_filter, 1000000)
#     # book.print_records()
# #     db_client = pymongo.MongoClient('localhost', 27017)
# #     book.save_to_db(db_client['nyse'])
#         
#     for list in book.symbols_dict.itervalues():
#         book.pickle_to_file(list[0].Symbol)
    
    book.symbols_dict = {}
    book.pickle_from_file('AIG')
    # db.test.aggregate({$group: {_id : "$symbol", count: {$sum : 1}}}, { $sort: {count: -1} });
    # book.read_from_db(db_client['nyse'], {'symbol': 'AZN'})
    # book.print_records()
    order_book = NyseOrderBook("AIG")
    for list in book.symbols_dict.itervalues():
        for order in list:
            order_book.process_order(order)
    book.print_records()
    prepare_data(order_book, 100)

if __name__ == '__main__':
    main()