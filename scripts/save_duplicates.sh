TOPIC=$1;
CONCEPT_TYPE=$2;
python3 image_finder.py ./images/"$TOPIC"/"$CONCEPT_TYPE"/ | tee duplicates.txt