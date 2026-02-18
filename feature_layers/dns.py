def extract_dns_features(url):
    return {"has_A_record": 1, "has_MX_record": 1, "num_nameservers": 2, "dns_ttl": 3600}
