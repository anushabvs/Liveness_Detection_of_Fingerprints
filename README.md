# Liveness_Detection_of_Fingerprints


Use of biometric authentication systems has increased incredibly in recent years in fields of security due to its permanence and uniqueness. However, it is also vulnerable to the certain type of attacks including presenting fake fingerprints to the sensor which requires the development of new and efficient protection measures.

In this work, a method to provide fingerprint vitality authentication, in order to improve vulnerability of fingerprint identification systems to spoofing is introduced. The method aims at detecting ‘liveness' in finger- print scanners by using the physiological phenomenon. A wavelet based approach is used which concentrates on the changing coefficients using the zoom-in property of the wavelets. Multi resolution analysis and wavelet analysis are used to extract information from low frequency and high frequency content of the images respectively. A variety of wavelets like Daubechies, Symlet, Coiflet wavelet etc. are designed and
implemented to perform the wavelet analysis. A threshold is applied to the first difference of the information in all the sub-bands. The energy content of the changing coefficients is used as a quantified measure to perform the desired classification, as they reflect a varying textural pattern.

The proposed algorithm was applied to the training data set i.e. the LivDet fingerprint dataset and was able to classify ‘live’ fingers from ‘fake’ fingers with an accuracy of ~92% , thus providing a method for enhanced security and improved spoof protection.
