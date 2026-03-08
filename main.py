def combine(email_score, url_score):

    final_score = 0.5 * email_score + 0.5 * url_score

    if final_score > 0.5:
        return "Phishing"
    else:
        return "Legitimate"