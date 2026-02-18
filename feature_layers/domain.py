import tldextract
def extract_domain_features(url):
    ext = tldextract.extract(url)
    return {"domain_age": -1, "domain_entropy": 3.5, "is_suspicious_tld": 0, "subdomain_depth": len(ext.subdomain.split(".")) if ext.subdomain else 0}
