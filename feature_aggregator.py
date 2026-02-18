from feature_layers.lexical import extract_lexical_features
from feature_layers.domain import extract_domain_features
from feature_layers.dns import extract_dns_features
from feature_layers.ssl_layer import extract_ssl_features
from feature_layers.reputation import extract_reputation_features

class FeatureAggregator:
    def extract_all_features(self, url):
        res = {}
        res.update(extract_lexical_features(url))
        res.update(extract_domain_features(url))
        res.update(extract_dns_features(url))
        res.update(extract_ssl_features(url))
        res.update(extract_reputation_features(url))
        return res
