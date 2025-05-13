import requests
import time
import csv
from datetime import datetime

# Data storage
crypto_data = []
known_wallets = {
    "bitcoin": ["Coinbase Wallet", "MetaMask", "Exodus", "Ledger", "Trezor"],
    "ethereum": ["Coinbase Wallet", "MetaMask", "Exodus", "Ledger", "Trezor"],
    "binancecoin": ["Trust Wallet", "MetaMask", "Ledger", "Trezor"],
    "solana": ["Phantom", "Coinbase Wallet", "Ledger", "Trezor"],
    "xrp": ["Ledger", "Trezor", "Exodus"]
}
known_exchanges = {
    "bitcoin": ["Coinbase", "Binance", "Kraken", "Bitstamp"],
    "ethereum": ["Coinbase", "Binance", "Kraken", "Bitstamp"],
    "binancecoin": ["Binance", "Coinbase", "Kraken"],
    "solana": ["Coinbase", "Binance", "Kraken"],
    "xrp": ["Coinbase", "Binance", "Bitstamp"]
}

# Simulated wallet reactions for Americans
wallet_reactions = [
    "Security Concerns: 36% of American crypto owners fear cyberattacks or losing wallet access.",
    "Adoption Trends: 28% of U.S. adults (65M) own crypto; 67% of owners plan to buy more in 2025.",
    "Political Impact: 60% believe crypto values will rise under Trump, but 40% doubt safety.",
    "Usability Issues: Many find wallets hard to use; Mark Cuban calls them 'awful'.",
    "Scam Fears: High concern over scams like Trump's $TRUMP coin, where 764,000 wallets lost money."
]

# U.S. crypto wallet regulations (as of May 2025)
us_regulations = [
    "Tax Reporting: IRS requires all crypto transactions to be reported; Form 1099-DA mandatory for wallet providers in 2025.",
    "AML/KYC Rules: FinCEN mandates reporting transactions over $10,000, including unhosted wallets, impacting privacy.",
    "Trump Policies: Pro-crypto push with Bitcoin Strategic Reserve, but stricter oversight to curb fraud.",
    "State Laws: NY's BitLicense limits wallet providers; some restrict services in certain states.",
    "User Sentiment: 63% of Americans lack confidence in crypto safety, partly due to regulatory fears."
]

# Replace with your CoinMarketCap API key
CMC_API_KEY = "your_coinmarketcap_api_key_here"

# Fetch data from CoinMarketCap API
def get_cmc_data():
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/gainers-losers"
    headers = {
        "X-CMC_PRO_API_KEY": CMC_API_KEY,
        "Accept": "application/json"
    }
    params = {
        "start": "1",
        "limit": "5",
        "timeframe": "24h",
        "convert": "USD"
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "data" not in data or "24h" not in data["data"]:
            return []
        
        new_data = []
        for coin in data["data"]["24h"]:
            name = coin["name"].lower()
            price = float(coin["quotes"]["USD"]["price"])
            change_24h = float(coin["quotes"]["USD"]["percent_change_24h"])
            reason = f"High trading volume and market interest (data from CoinMarketCap)"
            wallets = known_wallets.get(name, ["Coinbase Wallet", "Ledger"])
            exchanges = known_exchanges.get(name, ["Coinbase", "Binance"])
            new_data.append({
                "name": coin["name"],
                "price": price,
                "change_24h": change_24h,
                "reason": reason,
                "wallets": wallets,
                "exchanges": exchanges,
                "source": "CoinMarketCap"
            })
        return new_data
    except Exception as e:
        print(f"Error fetching CoinMarketCap data: {e}")
        return []

# Fetch data from CoinGecko API
def get_coingecko_data():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "price_change_percentage_24h_desc",
        "per_page": "5",
        "page": "1",
        "sparkline": "false"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        new_data = []
        for coin in data:
            name = coin["id"].lower()
            price = float(coin["current_price"])
            change_24h = float(coin["price_change_percentage_24h"])
            reason = f"High trading volume and market interest (data from CoinGecko)"
            wallets = known_wallets.get(name, ["Coinbase Wallet", "Ledger"])
            exchanges = known_exchanges.get(name, ["Coinbase", "Binance"])
            new_data.append({
                "name": coin["name"],
                "price": price,
                "change_24h": change_24h,
                "reason": reason,
                "wallets": wallets,
                "exchanges": exchanges,
                "source": "CoinGecko"
            })
        return new_data
    except Exception as e:
        print(f"Error fetching CoinGecko data: {e}")
        return []

# Combine data from both sources
def get_top_cryptos():
    cmc_data = get_cmc_data()
    coingecko_data = get_coingecko_data()
    
    combined_data = {}
    for entry in cmc_data:
        combined_data[entry["name"].lower()] = entry
    for entry in coingecko_data:
        if entry["name"].lower() not in combined_data:
            combined_data[entry["name"].lower()] = entry
    
    sorted_data = sorted(combined_data.values(), key=lambda x: x["change_24h"], reverse=True)[:5]
    return sorted_data

# Save data to CSV with all details
def save_to_csv(data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = False
    try:
        with open("top_cryptos_data.csv", "r") as file:
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    with open("top_cryptos_data.csv", "a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Name", "Price", "24h Change", "Reason", "Wallets", "Exchanges", "Source", "Wallet Reactions", "U.S. Regulations"])
        for crypto in data:
            row = [
                timestamp,
                crypto["name"],
                crypto["price"],
                crypto["change_24h"],
                crypto["reason"],
                ", ".join(crypto["wallets"]),
                ", ".join(crypto["exchanges"]),
                crypto["source"],
                "; ".join(wallet_reactions),
                "; ".join(us_regulations)
            ]
            writer.writerow(row)

# Save data to a text file for content creation
def save_to_text(data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"crypto_content_{timestamp}.txt"
    with open(filename, "w") as file:
        file.write("Top 5 Cryptocurrencies (24h Performance)\n")
        file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for i, crypto in enumerate(data, 1):
            file.write(f"{i}. {crypto['name']}\n")
            file.write(f"Price: ${crypto['price']:.2f}\n")
            file.write(f"24h Change: {crypto['change_24h']:.2f}% ({crypto['source']})\n")
            file.write(f"Reason: {crypto['reason']}\n")
            file.write(f"Wallets: {', '.join(crypto['wallets'])}\n")
            file.write(f"Exchanges: {', '.join(crypto['exchanges'])}\n")
            file.write("\n")
        file.write("Wallet Reactions for Americans:\n")
        for reaction in wallet_reactions:
            file.write(f"- {reaction}\n")
        file.write("\nU.S. Wallet Regulations:\n")
        for regulation in us_regulations:
            file.write(f"- {regulation}\n")
        file.write("\n---\nUse this data for blog posts, social media, or reports!\n")
    print(f"Content saved to {filename}")

# Display data in text format (console)
def display_text(data):
    print("\nTop 5 Cryptocurrencies (24h Performance)")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if not data:
        print("No data available. Fetching data...")
    else:
        for i, crypto in enumerate(data, 1):
            print(f"{i}. {crypto['name']}")
            print(f"Price: ${crypto['price']:.2f}")
            print(f"24h Change: {crypto['change_24h']:.2f}% ({crypto['source']})")
            print(f"Reason: {crypto['reason']}")
            print(f"Wallets: {', '.join(crypto['wallets'])}")
            print(f"Exchanges: {', '.join(crypto['exchanges'])}\n")
        
        print("Wallet Reactions for Americans:")
        for reaction in wallet_reactions:
            print(f"- {reaction}")
        
        print("\nU.S. Wallet Regulations:")
        for regulation in us_regulations:
            print(f"- {regulation}")
    print("\n" + "="*50)

# Main loop
running = True
last_refresh = time.time()
refresh_interval = 60  # Refresh every 60 seconds

while running:
    current_time = time.time()
    
    # Auto-refresh
    if current_time - last_refresh > refresh_interval:
        crypto_data = get_top_cryptos()
        if crypto_data:
            save_to_csv(crypto_data)
            save_to_text(crypto_data)
        display_text(crypto_data)
        last_refresh = current_time
    
    # Simulate a basic loop (press Ctrl+C to exit)
    try:
        time.sleep(1)  # Sleep to prevent excessive CPU usage
    except KeyboardInterrupt:
        print("\nExiting...")
        running = False
