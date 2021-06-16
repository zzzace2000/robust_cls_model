mkdir cct

echo 'Download the cct datasets'
wget http://www.vision.caltech.edu/~sbeery/datasets/caltechcameratraps18/eccv_18_all_images_sm.tar.gz
echo 'Unzip. It will take a while...'
tar -xzf eccv_18_all_images_sm.tar.gz
mv eccv_18_all_images_sm cct/

echo 'Download the cct datasets'
wget http://www.vision.caltech.edu/~sbeery/datasets/caltechcameratraps18/eccv_18_annotations.tar.gz
echo 'Unzip. It will take a while...'
tar -xzf eccv_18_annotations.tar.gz

mv eccv_18_annotation_files cct/

echo 'Download the CCT CF CAGAN images'
FILE_ID='1N3xGwYAU-TXexSq2riYJaXWJHtUxmOGU'
DOWNLOADED_FILENAME='cct_cagan.tar.gz'
curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > ${DOWNLOADED_FILENAME}
echo 'Unzip. It will take a while...'
tar -xzf ${DOWNLOADED_FILENAME}

mv cagan cct/
