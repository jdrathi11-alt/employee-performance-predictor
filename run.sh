#!/bin/bash
echo "Employee Performance Predictor - Quick Start"
echo "========================================="
echo "1. Installing dependencies..."
pip install pandas numpy matplotlib seaborn --quiet
echo "2. Running full pipeline..."
python main.py
echo "3. Pipeline complete! Check outputs/ directory"
echo "   - Plots: outputs/plots/"
echo "   - Predictions: outputs/predictions.csv" 
echo "   - HR Report: outputs/hr_insights_report.txt"

