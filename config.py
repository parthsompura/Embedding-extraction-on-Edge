from normalizaton_thresholds import NormalizationThresholds

model_path = './use-models/universal-sentence-encoder_4'
model_version = "universal_sentence_encoder_4"

vector_size = 512
minimum_text_length = 20
selected_metric = "cosine"


#max distance between embedding and centroid to be even
#considered in calculations
cos_centroid_classify_threshold = 0.4

#threshold to find texts that are probably too far
cos_density_threshold = 0.65

#how many outliers should we try to find
remove_up_to = 3

#used for distances between texts - the smaller the closer
cos_norm_threshold = NormalizationThresholds(
    norm_max=100, norm_min=0,
    norm_lower_bound=0.3, norm_upper_bound=0.65
)

#used for distances between classes
cos_centroid_threshold = NormalizationThresholds(
    norm_max=100, norm_min=0,
    norm_lower_bound=0.4, norm_upper_bound=0.65
)

#used for distances between text embedding and classes
cos_classify_threshold = NormalizationThresholds(
    norm_max=100, norm_min=0,
    norm_lower_bound=0, norm_upper_bound=0.75
)