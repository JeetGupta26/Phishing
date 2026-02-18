def extract_ssl_features(url):
    return {"has_https": 1 if url.startswith("https") else 0, "cert_age_days": 100, "days_until_expiry": 200, "is_self_signed": 0}
