#! /usr/bin/env python
# XCoin API-call sample script (for Python 3.X)
#
# @author	btckorea
# @date	2017-04-11
#
#
# First, Build and install pycurl with the following commands::
# (if necessary, become root)
#
# https://pypi.python.org/pypi/pycurl/7.43.0#downloads
#
# tar xvfz pycurl-7.43.0.tar.gz
# cd pycurl-7.43.0
# python setup.py --libcurl-dll=libcurl.so install
# python setup.py --with-openssl install
# python setup.py install

import sys
from bithumb.xcoin_api_client import *
import pprint


api_key = "cd5168e51f7ba8fba4f17b94e4679cea"
api_secret = "8d54d4bebfaca83a6c87e4bb51d065eb"

api = XCoinAPI(api_key, api_secret)

rgParams = {
	"group_orders" : 1, # 0 or 1
	"count" : 50, # 1 ~ 50
	"order_currency" : "ETH",
	"payment_currency" : "KRW"
}


#
# public api
#
# /public/ticker
# /public/recent_ticker
# /public/orderbook
# /public/recent_transactions
while True:
	time.sleep(0.5)
	result = api.xcoinApiCall(api.params.public.orderbook, rgParams)
	print(result["data"])
	#print("status: " + result["status"])
	#print("last: " + result["data"]["closing_price"])
	#print("sell: " + result["data"]["sell_price"])
	#print("buy: " + result["data"]["buy_price"])
	#print("usec time: " + api.usecTime() + ", msc time: " + api.microtime())

#
# private api
#
# endpoint		=> parameters
# /info/current
# /info/account
# /info/balance
# /info/wallet_address

#result = api.xcoinApiCall(api.params.private.account, rgParams)
#print("status: " + result["status"])
#print("created: " + result["data"]["created"])
#print("account id: " + result["data"]["account_id"])
#print("trade fee: " + result["data"]["trade_fee"])
#print("balance: " + result["data"]["balance"])


