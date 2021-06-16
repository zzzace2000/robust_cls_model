mkdir waterbird

echo 'Download the waterbird'
wget https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz
echo 'Unzip. It will take a while...'
tar -xzf waterbird_complete95_forest2water2.tar.gz

mv waterbird_complete95_forest2water2 waterbird/

echo 'Download the waterbird segmentations'
FILE_ID='1EU6jUDf8tg4cIAJiv8lSEsfQhPe9rCwt'
DOWNLOADED_FILENAME='waterbird_segmentations.tar.gz'
curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > ${DOWNLOADED_FILENAME}
echo 'Unzip. It will take a while...'
tar -xzf ${DOWNLOADED_FILENAME}

mv segmentations waterbird/

echo 'Download the waterbird cagan'
FILE_ID='1MZQ6IAYmwnUfaIU8KLeh6ZlLb6VS5Awk'
DOWNLOADED_FILENAME='waterbird_cagan.tar.gz'
curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > ${DOWNLOADED_FILENAME}
echo 'Unzip. It will take a while...'
tar -xzf ${DOWNLOADED_FILENAME}

mv cagan waterbird/