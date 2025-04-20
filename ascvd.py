#!/usr/bin/env python3
"""
ASCVD Risk Calculator Module

This module contains functions to calculate ASCVD points,
convert those points to a risk estimate, and combine with additional scores.
"""

def calculate_ascvd_points(age, sex, total_chol, hdl, systolic_bp, bp_treatment, smoker, diabetic):
    points = 0

    # Age: only score if both age and sex are provided
    if age is not None and sex is not None:
        if sex.lower() == "male":
            if 40 <= age < 45: points += 0
            elif 45 <= age < 50: points += 3
            elif 50 <= age < 55: points += 6
            elif 55 <= age < 60: points += 8
            elif 60 <= age < 65: points += 10
            elif 65 <= age < 70: points += 11
            elif 70 <= age < 75: points += 12
            elif 75 <= age < 80: points += 13
        elif sex.lower() == "female":
            if 40 <= age < 45: points += -7
            elif 45 <= age < 50: points += -3
            elif 50 <= age < 55: points += 0
            elif 55 <= age < 60: points += 3
            elif 60 <= age < 65: points += 6
            elif 65 <= age < 70: points += 8
            elif 70 <= age < 75: points += 10
            elif 75 <= age < 80: points += 12

    # Total Cholesterol
    if total_chol is not None and sex is not None:
        if sex.lower() == "male":
            if total_chol < 160: points += 0
            elif 160 <= total_chol < 200: points += 4
            elif 200 <= total_chol < 240: points += 7
            elif 240 <= total_chol < 280: points += 9
            elif total_chol >= 280: points += 11
        elif sex.lower() == "female":
            if total_chol < 160: points += 0
            elif 160 <= total_chol < 200: points += 4
            elif 200 <= total_chol < 240: points += 8
            elif 240 <= total_chol < 280: points += 11
            elif total_chol >= 280: points += 13

    # HDL Cholesterol
    if hdl is not None:
        if hdl >= 60: 
            points += -1
        elif 50 <= hdl < 60: 
            points += 0
        elif 40 <= hdl < 50: 
            points += 1
        elif hdl < 40: 
            points += 2

    # Systolic Blood Pressure and Treatment
    if systolic_bp is not None and bp_treatment is not None:
        if bp_treatment:
            if systolic_bp < 120: points += 0
            elif 120 <= systolic_bp < 130: points += 3
            elif 130 <= systolic_bp < 140: points += 4
            elif 140 <= systolic_bp < 160: points += 5
            elif systolic_bp >= 160: points += 6
        else:
            if systolic_bp < 120: points += 0
            elif 120 <= systolic_bp < 130: points += 1
            elif 130 <= systolic_bp < 140: points += 2
            elif 140 <= systolic_bp < 160: points += 3
            elif systolic_bp >= 160: points += 4

    # Smoking status
    if smoker is not None and sex is not None:
        if smoker:
            points += 4 if sex.lower() == "male" else 3

    # Diabetes status
    if diabetic is not None and sex is not None:
        if diabetic:
            points += 2 if sex.lower() == "male" else 4

    return points

def points_to_risk(points, sex):
    # If sex is missing, return the points as a fallback
    if sex is None:
        return points
    if sex.lower() == "male":
        if points < 0: 
            return 1
        elif points <= 4: 
            return 1
        elif points == 5: 
            return 2
        elif points == 6: 
            return 2
        elif points == 7: 
            return 3
        elif points == 8: 
            return 4
        elif points == 9: 
            return 5
        elif points == 10: 
            return 6
        elif points == 11: 
            return 8
        elif points == 12: 
            return 10
        elif points == 13: 
            return 12
        elif points == 14: 
            return 16
        elif points == 15: 
            return 20
        elif points == 16: 
            return 25
        else: 
            return 30
    else:
        if points < 9: 
            return 1
        elif points <= 11: 
            return 1
        elif points == 12: 
            return 2
        elif points == 13: 
            return 2
        elif points == 14: 
            return 3
        elif points == 15: 
            return 4
        elif points == 16: 
            return 5
        elif points == 17: 
            return 6
        elif points == 18: 
            return 8
        elif points == 19: 
            return 11
        elif points == 20: 
            return 14
        elif points == 21: 
            return 17
        elif points == 22: 
            return 22
        else: 
            return 27

def calculate_ascvd_risk(age, sex, total_chol, hdl, systolic_bp, bp_treatment, smoker, diabetic, rcri=None, sts=None):
    """
    Calculates the base ASCVD risk from patient data using the ASCVD points system,
    then incorporates additional risk scores (RCRI and STS) if provided.
    
    Weighting:
      - Base ASCVD risk: 50%
      - RCRI: 25%
      - STS: 25%
      
    If additional scores are missing, weights are re-adjusted.
    Returns the final risk percentage, capped at 100%.
    """
    pts = calculate_ascvd_points(age, sex, total_chol, hdl, systolic_bp, bp_treatment, smoker, diabetic)
    base_risk = points_to_risk(pts, sex)
    
    total_weight = 0.5
    weighted_sum = 0.5 * base_risk
    if rcri is not None:
        weighted_sum += 0.25 * rcri
        total_weight += 0.25
    if sts is not None:
        weighted_sum += 0.25 * sts
        total_weight += 0.25
        
    final_risk = weighted_sum / total_weight
    final_risk = round(final_risk, 2)
    return min(final_risk, 100)

if __name__ == "__main__":
    # Basic test example
    risk = calculate_ascvd_risk(55, "male", 210, 45, 130, True, False, True)
    print("Calculated risk:", risk)
