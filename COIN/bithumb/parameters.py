class Currency:
    def __init__(self):
        # BTC, ETH, DASH, LTC, ETC, XRP, BCH, XMR, ZEC, QTUM, BTG, EOS (기본값: BTC), ALL(전체)
        self.BTC = "BTC"
        self.ETH = "ETH"
        self.DASH = "DASH"
        self.LTC = "LTC"
        self.ETC = "ETC"
        self.XRP = "XRP"
        self.BCH = "BCH"
        self.XMR = "XMR"
        self.ZEC = "ZEC"
        self.QTUM = "QTUM"
        self.BTG = "BTG"
        self.EOS = "EOS"

class Private:
    def __init__(self):
        self.account="/info/account"  # bithumb 거래소 회원 정보
        self.balance="/info/balance"  # bithumb 거래소 회원 지갑 정보
        self.wallet_addr="/info/wallet_address"  # bithumb 거래소 회원 입금 주소
        self.ticker="/info/ticker"  # 회원 마지막 거래 정보
        self.orders="/info/orders"  # 판/구매 거래 주문 등록 또는 진행 중인 거래
        self.user_trans="/info/user_transactions"  # 회원 거래 내역
        self.trade_place="/trade/place"  # bithumb 회원 판/구매 거래 주문 등록 및 체결
        self.order_detail="/info/order_detail"  # bithumb 회원 판/구매 체결 내역
        self.cancel="/trade/cancel"  # bithumb 회원 판/구매 체결 취소
        self.btc_withdrawal="/trade/btc_withdrawal"  # bithumb 회원 Currency 출금
        self.krw_deposit="/trade/krw_deposit"  # bithumb 회원 KRW 입금 가상계좌 정보 요청
        self.krw_withdrawal="/trade/krw_withdrawal"  # bithumb 회원 KRW 출금 신청
        self.market_buy="/trade/market_buy"  # 시장가 구매
        self.market_sell="/trade/market_sell"  # 시장가 판매


class Public:
    def __init__(self):
        self.ticker = "/public/ticker/"  # bithumb 거래소 마지막 거래 정보
        self.orderbook = "/public/orderbook/"  # bithumb 거래소 판/구매 등록 대기 또는 거래 중 내역 정보
        self.recent_transactions = "/public/recent_transactions/"  # bithumb 거래소 거래 체결 완료 내역

class Params:
    def __init__(self):
        self.public = Public()
        self.private = Private()
        self.currency = Currency()

