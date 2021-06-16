
echo 'Download the IN-9 test sets...'
wget https://github.com/MadryLab/backgrounds_challenge/releases/download/data/backgrounds_challenge_data.tar.gz
echo 'Unzip. It will take a while...'
tar -xzf backgrounds_challenge_data.tar.gz

echo 'Move the data into the bg_challenge/test/'
mv bg_challenge test
mkdir bg_challenge
mv test bg_challenge/

mkdir bg_challenge/train

echo 'Download and extract original datasets'
wget -O original.tar.gz https://www.dropbox.com/s/0vv2qsc4ywb4z5v/original.tar.gz?dl=0
tar -xzf original.tar.gz
mv original bg_challenge/train/

echo 'Download and extract Mixed-Rand datasets...'
wget -O mixed_rand.tar.gz https://www.dropbox.com/s/cto15ceadgraur2/mixed_rand.tar.gz?dl=0
tar -xzf mixed_rand.tar.gz
mv mixed_rand bg_challenge/train/

echo 'Download and extract no_fg datasets'
wget -O no_fg.tar.gz https://www.dropbox.com/s/0v6w9k7q7i1ytvr/no_fg.tar.gz?dl=0
tar -xzf no_fg.tar.gz
mv no_fg bg_challenge/train/

echo 'Download and extract mixed_next datasets'
wget -O mixed_next.tar.gz https://www.dropbox.com/s/4hnkbvxastpcgz2/mixed_next.tar.gz?dl=0
tar -xzf mixed_next.tar.gz
mv mixed_next bg_challenge/train/

echo 'Download and extract mixed_same datasets'
wget -O mixed_same.tar.gz https://www.dropbox.com/s/f2525w5aqq67kk0/mixed_same.tar.gz?dl=0
tar -xzf mixed_same.tar.gz
mv mixed_same bg_challenge/train/


echo 'Download and extract cagan cf (original_bbox_cf_cagan) datasets'
FILE_ID='1kiBz3E8bFTPn_S9-jshtu3jjqMGReCvK'
DOWNLOADED_FILENAME='original_bbox_cf_cagan.tar.gz'
curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > ${DOWNLOADED_FILENAME}
echo 'Unzip. It will take a while...'
tar -xzf ${DOWNLOADED_FILENAME}

mv original_bbox_cf_cagan bg_challenge/train/

#echo 'Download and extract in9l datasets'
#wget -O in9l.tar.gz https://www.dropbox.com/s/8w29bg9niya19rn/in9l.tar.gz?dl=0
#tar -xzf in9l.tar.gz
#mv in9l bg_challenge/train/


