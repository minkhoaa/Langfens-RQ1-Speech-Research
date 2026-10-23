L2-ARCTIC documentation – PSI Lab
Skip to content
Home
About
Research
Publications
People
Teaching
Contact
MobileStress
Site Search
Search for:
Search
L2-ARCTIC documentation
Summary
The L2-ARCTIC corpus is a speech corpus of non-native English that is intended for research in voice conversion, accent conversion, and mispronunciation detection. In total, the corpus contains 26,867 utterances from 24 non-native speakers with a balanced gender and L1 distribution. Most speakers recorded the full CMU ARCTIC set. The total duration of the corpus is 27.1 hours, with an average of 67.7 minutes (std: 8.6 minutes) of speech per L2 speaker. On average, each utterance is 3.6 seconds in duration. The pause before and after each utterance is generally no longer than 100 ms. Using the forced alignment results, we estimate a speech to silence ratio of 7:1 across the whole dataset. The dataset contains over 238,702 word segments, giving an average of around nine (9) words per utterance, and over 851,830 phone segments (excluding silence).
Human annotators manually examined 3,599 utterances, annotating 14,098 phone substitutions, 3,420 phone deletions, and 1,092 phone additions.
Some speakers did not read all sentences, and a few sentences were removed for some speakers since those recordings did not have the required quality.
About the suitcase corpus (added on March 12, 2020)
This portion of the L2-ARCTIC corpus involves spontaneous speech. We include recordings and annotations from 22 of the 24 speakers who recorded the sentences. Speakers SKA and ASI did not participate in this task. Each speaker retold a story from a picture narrative used in applied linguistics research on comprehensibility, accentedness, and intelligibility. The pictures are generally known as the suitcase story. Each retelling of the narrative was done after looking over the story and asking the researchers questions about what was happening. Few participants had questions. The annotations were carried out by two research assistants trained in phonetic transcription. Each did half of the transcriptions, then checked the other half done by the other research assistant. Finally, all transcriptions were checked by John Levis, a co-PI for the project. This project was funded by National Science Foundation award 1618953, titled “Developing Golden Speakers for Second-Language Pronunciation.”
The total duration of this subset is 26.1 minutes, with an average of 1.2 minutes (std: 41.5 seconds) per speaker. Using the manual annotation results, we estimate a speech to silence ratio of 2.3:1 across the whole dataset. The dataset contains around 3,083 word segments, giving an average of 140 words per recording, and around 9,458 phone segments (excluding silence). The manual annotations include 1,673 phone substitutions, 456 phone deletions, and 90 phone additions.
Manual annotation
Each manually annotated TextGrid file will always have a “words” and “phones” tier, but some files may have an additional tier that contains comments from the annotators. Each phone segment was tagged in the “phones” tier following the conventions below:
Correctly pronounced: the existing forced-alignment label was unchanged
Phone substitution error: We changed the label to “CPL,PPL,s”, where “CPL” is the correct phoneme label (i.e., what should have been produced), “PPL” is the perceived phoneme label (i.e., what was actually produced), and “s” stands for substitution. If the perceived phoneme label was hard to judge, we used the “err” tag as its phoneme label, i.e., tagged it as “CPL,err,s”. If the perceived phoneme sounded like a deviation of the standard American English pronunciation, we marked it using a “deviation” symbol “*”. For example, if the correct label for a phone segment is “AH”, and the speaker pronounced it as an “AO” with a foreign accent, we mark this error as “AH,AO*,s”
Phone addition error: we created an empty interval in the “phones” tier, adjusted its boundaries, and changed its label to “sil,PPL,a”, where “sil” stands for silence, and “a” stands for addition (insertion). If the perceived phoneme label was not in the American English phoneme set, we used the “err” and “*” tags as noted above
Phone deletion error: we found a silent segment where that phone segment should be, and annotated it as “CPL,sil,d”, where “d” stands for deletion
The “phones” tier will only contain the ARPAbet symbols while the comments may contain IPA symbols (in UTF8 format).
Directory structure
Data for each speaker is organized in its own subdirectory under the root folder; the root folder also contains a README.md file, a README.pdf file (the README.md converted to PDF), a LICENSE file, and a PROMPTS file (containing the original text prompts from CMU ARCTIC). Each speaker’s directory is structured as follows:
/wav: containing audio files in WAV format, sampled at 44.1 kHz
/transcript: containing orthographic transcriptions, saved in TXT format
/textgrid: containing phoneme transcriptions generated from forced-alignment, saved in TextGrid format
/annotation: containing manual annotations, saved in TextGrid format
The suitcase corpus is stored in a subfolder named suitcase_corpus under the root folder and follows a similar directory structure, except that we did not include the forced-aligned TextGrid files, since the manual annotations contain more accurate alignments. All files in the suitcase corpus are named by speaker codes.
File summary
Speaker
# Wav Files
# Annotations
ABA
1129
150
SKA
974
150
YBAA
1130
149
ZHAA
1132
150
BWC
1130
150
LXC
1131
150
NCC
1131
150
TXHC
1132
150
ASI
1131
150
RRBI
1130
150
SVBI
1132
150
TNI
1131
150
HJK
1131
150
HKK
1131
150
YDCK
1131
150
YKWK
1131
150
EBVS
1007
150
ERMS
1132
150
MBMPS
1132
150
NJS
1131
150
HQTV
1132
150
PNV
1132
150
THV
1132
150
TLV
1132
150
Total
26867
3599
Phoneme set
Helpful toolkits
Below are some useful tools we used to access TextGrid files,
Praat: visualizing and modifying TextGrid files in a GUI
mPraat: read/write TextGrid files in Matlab
TextGridTools: read/write TextGrid files in Python
Revision history
03/12/2020: v5.0, add the suitcase corpus, which contains un-scripted speech and corresponding annotations from 22 of the 24 speakers
06/06/2019: v4.0 is available. We re-examined some of the annotations and changed most of the “err” tags to more detailed (and informative) annotations — marking them as different deviations from standard English
04/08/2019: v3.0, add 4 Vietnamese speakers to the corpus
09/28/2018: v2.0, add 10 new speakers to the corpus
03/26/2018: v1.0, the initial release
Disclaimer
We may recommend the use of software, information, products, or websites that are owned or operated by other parties. We offer or facilitate this recommendation by hyperlinks or other methods to aid your access to the third-party resource. While we endeavor to direct you to helpful, trustworthy resources, we cannot endorse, approve, or guarantee software, information, products, or services provided by or at a third-party resource or track changes in the resource. Thus, we are not responsible for the content or accuracy of any third-party resource or for any loss or damage of any sort resulting from the use of, or for any failure of, products or services provided at or from a third party resource. We recommend these resources on an “as is” basis. When you use a third-party resource, you will be subject to its terms and licenses and no longer be protected by our privacy policy or security practices, which may differ from the third policy or practices or other terms. You should familiarize yourself with any license or use terms of, and the privacy policy and security practices of, the third party resource, which will govern your use of that resource.
Physical: 309 Peterson Building, Texas A&M University
Mailing: PSI Lab, 3112 TAMU, College Station, TX 77843
Theme by Colorlib Powered by WordPress
Back to top
Loading Comments...
Write a Comment...
Email (Required)
Name (Required)
Website