      SUBROUTINE EDFTST(UNIFRM,N,UNIT3)
C
C     THIS SUBROUTINE CALCULATES STATISTICS WHICH TEST THE ASSUMPTION
C     THAT THE PASSED I.I.D. UNIFORM(0,1) DEVIATES ARE IN FACT FROM SUCH
C     A DISTRIBUTION.  THE STATISTICS ARE THE CRAMER-VON MISES W SQUARED,
C     THE WATSON U SQUARED, THE ANDERSON-DARLING A SQUARED, AND THE
C     KOLMORGOROV-SMIRNOV D+, D-, AND D.  SEE THE REFERENCE:
C     STEPHENS, M.A., 'EDF STATISTICS FOR GOODNESS OF FIT AND
C     SOME COMPARISONS', JASA V.69, PP. 730-737, SEPTEMBER, 1974.
C     THE SUBROUTINE ALSO CALCULATES SOME BINOMIAL PROBABILITIES FOR
C     DETECTING ABNORMAL CLUSTERING OF THE UNIFORM DEVIATES NEAR ZERO.
C     NOTE THAT THE INPUT ARRAY UNIFRM CONTAINING THE DEVIATES WILL BE
C     DESTROYED IN THE PROCESS OF COMPUTING THE BINOMIAL PROBABILITIES.
C
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      DOUBLE PRECISION ALPHA(3),UNIFRM(N)
      INTEGER COUNT(3),UNIT3
      SAVE ALPHA
      DATA ALPHA/0.10D0,0.05D0,0.01D0/
C
      N1=N+1
      ANIN=1.D0/N
      TWONIN=0.5D0/N
      ROOTN=SQRT(DBLE(N))
      ZSUM=0.D0
      A2SUM=0.D0
      W2SUM=0.D0
      IANDER=0
      DPLUS=0.D0
      DMINUS=0.D0
C
C     BEGIN BY SORTING THE SUPPOSED UNIFORM(0,1) VARIATES.
C
      CALL HPSORT(UNIFRM,N)
C
C     CALCULATE THE SUMS REQUIRED FOR THE STATISTICS.  IF A VALUE OF
C     ZERO OR ONE IS ENCOUNTERED, THE ANDERSON-DARLING STATISTIC CANNOT
C     BE CALCULATED.
C
      DO 10 I=1,N
      UNI=UNIFRM(I)
      DPLUS=MAX(I*ANIN-UNI,DPLUS)
      DMINUS=MAX(UNI-(I-1)*ANIN,DMINUS)
      ZSUM=ZSUM+UNI
      W2SUM=W2SUM+(UNI-(I+I-1)*TWONIN)**2
      UNII=UNIFRM(N1-I)
      IF (UNI.GT.0.D0.AND.UNII.LT.1.D0) THEN
      A2SUM=A2SUM+(I+I-1)*(DLOG(UNI)+DLOG(1.0D0-UNII))
      ELSE
      IANDER=1
      END IF
 10   CONTINUE
C
C     COMPUTE THE EDF STATISTICS.
C
      CRAMER=W2SUM+TWONIN/6.D0
      ZSUM=ZSUM/N-0.5D0
      WATSON=CRAMER-N*ZSUM*ZSUM
      ANDER=-A2SUM/N-N
      DKS=MAX(DPLUS,DMINUS)
      VKUPER=DPLUS+DMINUS
C
C     CALCULATE THE SAMPLE-SIZE-MODIFIED VALUES FOR THE EDF STATISTICS.
C
      WATMOD=(WATSON-0.1D0*ANIN+0.1D0*ANIN*ANIN)*(1.D0+0.8D0*ANIN)
      CRAMOD=(CRAMER-0.4D0*ANIN+0.6D0*ANIN*ANIN)*(1.D0+ANIN)
      DPMOD=DPLUS*(ROOTN+0.12D0+0.11D0/ROOTN)
      DMMOD=DMINUS*(ROOTN+0.12D0+0.11D0/ROOTN)
      DKMOD=DKS*(ROOTN+0.12D0+0.11D0/ROOTN)
      VKMOD=VKUPER*(ROOTN+0.155D0+0.24D0/ROOTN)
C
C     OUTPUT THE STATISTIC HEADER.
C
      WRITE(UNIT3,20)
 20   FORMAT(' IF THE CURRENT MODEL IS VALID, THESE DEVIATES SHOULD'
     1/' REPRESENT A RANDOM SAMPLE FROM A UNIFORM DISTRIBUTION ON'
     2/' THE INTERVAL (0,1).  THE FOLLOWING STATISTICS USE THE'
     3/' EMPIRICAL DISTRIBUTION FUNCTION OF THE DEVIATES TO TEST FOR'
     4/' DEPARTURES FROM THE UNIFORM DISTRIBUTION.  SEE THE PAPER:'
     5/' M.A. STEPHENS, EDF STATISTICS FOR GOODNESS OF FIT AND SOME'
     6/' COMPARISONS. JASA, VOL.69, PP.730-737, 1974.'/)
C
C     OUTPUT THE PVALUE TABLE.
C
      WRITE(UNIT3,30)
 30   FORMAT(' UPPER PERCENTAGE POINTS FOR THE MODIFIED STATISTICS:'//
     1,'    STATISTIC                      PVALUES'/
     2 '                       .10      .05      .025     .01'//
     3,' ANDERSON-DARLING     1.933    2.492     3.020   3.857'/
     4,' CRAMER-VON MISES     0.347    0.461     0.581   0.743'/
     5,'      WATSON          0.152    0.187     0.221   0.267'/
     6,'        D+            1.073    1.224     1.358   1.518'/
     7,'        D-            1.073    1.224     1.358   1.518'/
     8,'        D             1.224    1.358     1.480   1.628'/
     9,'      KUIPER          1.620    1.747     1.862   2.001'/)
C
C     OUTPUT THE STATISTICS.
C
      WRITE(UNIT3,40) ANDER,ANDER,CRAMER,CRAMOD,WATSON,WATMOD
     1,DPLUS,DPMOD,DMINUS,DMMOD,DKS,DKMOD,VKUPER,VKMOD
 40   FORMAT(' SAMPLE VALUES FOR THESE DATA:'//'    STATISTIC'
     1,10X,'VALUE    MODIFIED VALUE'//
     2' ANDERSON-DARLING   ',F8.3,6X,F8.3/
     3' CRAMER-VON MISES   ',F8.3,6X,F8.3/
     4'     WATSON         ',F8.3,6X,F8.3/
     5'       D+           ',F8.3,6X,F8.3/
     6'       D-           ',F8.3,6X,F8.3/
     7'       D            ',F8.3,6X,F8.3/
     8'     KUIPER         ',F8.3,6X,F8.3)
C
      IF (N.LE.5) WRITE(UNIT3,50)
 50   FORMAT(/' *** WARNING ***  EDF STATISTIC PVALUES ARE NOT ACCURATE'
     1/' WHEN THE NUMBER OF DEVIATES IS SMALL.')
      IF (IANDER.EQ.1) WRITE(UNIT3,60)
 60   FORMAT(/' *** WARNING ***  ANDERSON-DARLING STATISTIC IS AT BEST'
     1/' ONLY APPROXIMATELY CORRECT DUE TO THE PRESENCE OF DEVIATES'
     2/' OF ZERO OR ONE.')
C
C     CALCULATE THE BINOMIAL PROBABILITY ASSOCIATED WITH THE OBSERVED
C     NUMBERS OF SMALL VALUES.  FIRST OUTPUT A HEADER.
C
      WRITE(UNIT3,70)
 70   FORMAT(//' THE BINOMIAL STATISTIC ASSOCIATED WITH THE OBSERVED'
     1/' NUMBERS OF DEVIATES LESS THAN ALPHA, ALPHA = .10, .05 AND'
     2/' .01., GIVES A TEST FOR AN OVER-ABUNDANCE OF SMALL DEVIATES.'
     3//,' ALPHA     NUMBER         PVALUE'/)
C
C     COUNT THE NUMBER OF SMALL DEVIATES.
C
      DO 80 I=1,3
      A=ALPHA(I)
      DO 90 J=1,N
 90   IF (UNIFRM(J).GT.A) GO TO 80
      J=N+1
 80   COUNT(I)=J-1
C
C     COMPUTE THE PVALUES ASSOCIATED WITH HAVING AT LEAST THESE
C     MANY SMALL DEVIATES.
C
      DO 100 I=1,3
      A=ALPHA(I)
      M=COUNT(I)
      IF (M.EQ.0) THEN
C
C     IF THERE ARE NO SUCH SMALL VALUES, THEN THE PVALUE IS ONE.
C
      PVALUE=1.0D0
      ELSE
C
C     CALCULATE THE REQUIRED BINOMIAL PROBABILITIES RECURSIVELY.  IMAGINE N
C     TRIALS.  AT THE KTH TRIAL A BERNOULLI EXPERIMENT IS CARRIED OUT WITH
C     SUCCESS PROBALBILITY ALPHA.  WE WRITE OVER UNIFRM SO THAT UNIFRM(L)
C     REPRESENTS THE PROBABILITY OF EXACTLY L-1 SUCCESSES IN THESE K TRIALS.
C     NOTE THAT AS SOON AS K IS AS LARGE AS M, ONLY THE PROBABILITIES
C     UNIFRM(1) THROUGH UNIFRM(M) NEED BE COMPUTED.
C
      A1=1.0D0-A
      UNIFRM(1)=1.0D0
      DO 110 K=1,N
      IF (K.LT.M) THEN
      UNIFRM(K+1)=A*UNIFRM(K)
      LSTART=K
      ELSE
      LSTART=M
      END IF
      DO 120 L=LSTART,2,-1
 120  UNIFRM(L)=A1*UNIFRM(L)+A*UNIFRM(L-1)
 110  UNIFRM(1)=A1*UNIFRM(1)
C
C     SUM TO FIND THE LEFT TAIL PROBABILITY AND SUBTRACT FROM ONE.
C
      SUM=0.D0
      DO 130 K=1,M
 130  SUM=SUM+UNIFRM(K)
      PVALUE=1.D0-SUM
      END IF
 100  WRITE(UNIT3,140) A,M,PVALUE
 140  FORMAT(2X,F4.2,I9,11X,F7.5)
      END
