

      SUBROUTINE CORRPEDCALCWRAPPERONE(SNPCOUNT,EPED,FREQS,
     / BUFFER,NSUBJECTS,ALPHA,ARRAYSIZE,MAXSNPCOLS)
      implicit none
      INTEGER NSUBJECTS,ARRAYSIZE,MAXSNPCOLS,SNPCOLS
      INTEGER SNPCOUNT(ARRAYSIZE),SNPCOL
      REAL ALPHA, EPED(ARRAYSIZE)
      CHARACTER BUFFER(NSUBJECTS,MAXSNPCOLS)
      INTEGER ROW,COL,ROWVAL,COLVAL,PEDINDEX
      REAL FREQS(MAXSNPCOLS),FREQ,VAR,NUMER
      SNPCOLS = 0
      CALL FILLBUFFERTHREAD(BUFFER,FREQS,SNPCOLS)
      
      
      DO WHILE ( SNPCOLS .NE. 0 )
  !    	CALL CORRPEDCALCONE(BUFFER,EPED,SNPCOUNT,NSUBJECTS                 
  !   /       ,SNPCOLS,ARRAYSIZE,ALPHA)
      do 10 SNPCOL = 1,SNPCOLS
      	FREQ = FREQS(SNPCOL)
      	IF (FREQ .EQ. -1.) GOTO 10
  	IF (FREQ .NE. 0.0 .AND. FREQ .NE. 1.0) THEN
  	   VAR = (2.*FREQ*(1.-FREQ))**ALPHA
  	   PEDINDEX = 1
  	   do 30 COL = 1,NSUBJECTS
  	      COLVAL = iachar(BUFFER(COL,SNPCOL))
  	      IF (COLVAL .NE. 3) THEN
  !		    PEDINDEX = 1 + ((COL-1)*MAPSIZE) - 
  !   /              ((COL-1)*((COL-1)-1)/2)
     		 NUMER = (COLVAL - 2.*FREQ)*VAR
     		 do 40 ROW = COL,NSUBJECTS
     		    ROWVAL = iachar(BUFFER(ROW,SNPCOL))
     		    IF (ROWVAL .NE. 3) THEN
     		       EPED(PEDINDEX) = EPED(PEDINDEX) +
     /                 NUMER*(ROWVAL-2.*FREQ)
     		       SNPCOUNT(PEDINDEX) = SNPCOUNT(PEDINDEX) + 1
     		    ENDIF
     		    PEDINDEX = PEDINDEX + 1
  40   		 continue
  	      ELSE
  		 PEDINDEX = PEDINDEX + NSUBJECTS - COL + 1
              ENDIF
  30       continue  
        ELSE
  	   PEDINDEX = 1
  	   do 50 COL = 1,NSUBJECTS
              IF (iachar(BUFFER(COL,SNPCOL)) .NE. 3) THEN
  !		    PEDINDEX = 1 + ((COL-1)*MAPSIZE) - 
  !   /              ((COL-1)*((COL-1)-1)/2)
     		    
     		 do 60 ROW = COL,NSUBJECTS
     		    IF (iachar(BUFFER(ROW,SNPCOL)) .NE. 3)
     / 		    THEN
     		       SNPCOUNT(PEDINDEX) = SNPCOUNT(PEDINDEX) + 1
     		    ENDIF
     		    PEDINDEX = PEDINDEX + 1
  60   		 continue
  	      ELSE
  		 PEDINDEX = PEDINDEX + NSUBJECTS - COL + 1
              ENDIF
  50       continue                   
        ENDIF
  10  continue  

        CALL FILLBUFFERTHREAD(BUFFER,FREQS,SNPCOLS)
       
      END DO
      RETURN
      END
      
      SUBROUTINE CORRPEDCALCONE(BUFFER,EPED,SNPCOUNT,NSUBJECTS,
     /SNPCOLS,ARRAYSIZE,ALPHA)
      implicit none
      INTEGER NSUBJECTS,ARRAYSIZE,SNPCOLS
      INTEGER SNPCOUNT(ARRAYSIZE)
      REAL ALPHA,EPED(ARRAYSIZE)
      CHARACTER BUFFER(NSUBJECTS,SNPCOLS)
      INTEGER SNPCOL,MISSINGSUBJECTS,FREQSUM
      INTEGER ROW,COL,ROWVAL,COLVAL,PEDINDEX
      REAL FREQ,VAR,NUMER
      
      do 10 SNPCOL = 1,SNPCOLS
      	FREQSUM = 0
      	MISSINGSUBJECTS = 0
      	do 20 ROW = 1,NSUBJECTS
      		ROWVAL = iachar(BUFFER(ROW,SNPCOL))
      		IF (ROWVAL .NE. 3) THEN
      			FREQSUM = FREQSUM + ROWVAL
      		ELSE
      			MISSINGSUBJECTS=MISSINGSUBJECTS+1
      		ENDIF
  20	continue
  	IF (MISSINGSUBJECTS .EQ. NSUBJECTS) goto 10
  	FREQ = FREQSUM
  	FREQ = FREQ/(2.*(NSUBJECTS-MISSINGSUBJECTS))
  	
  	IF (FREQ .NE. 0.0 .AND. FREQ .NE. 1.0) THEN
  	   VAR = (2.*FREQ*(1.-FREQ))**ALPHA
  	   PEDINDEX = 1
  	   do 30 COL = 1,NSUBJECTS
  	      COLVAL = iachar(BUFFER(COL,SNPCOL))
  	      IF (COLVAL .NE. 3) THEN
  !		    PEDINDEX = 1 + ((COL-1)*MAPSIZE) - 
  !   /              ((COL-1)*((COL-1)-1)/2)
     		 NUMER = (COLVAL - 2.*FREQ)*VAR
     		 do 40 ROW = COL,NSUBJECTS
     		    ROWVAL = iachar(BUFFER(ROW,SNPCOL))
     		    IF (ROWVAL .NE. 3) THEN
     		       EPED(PEDINDEX) = EPED(PEDINDEX) +
     /                 NUMER*(ROWVAL-2.*FREQ)
     		       SNPCOUNT(PEDINDEX) = SNPCOUNT(PEDINDEX) + 1
     		    ENDIF
     		    PEDINDEX = PEDINDEX + 1
  40   		 continue
  	      ELSE
  		 PEDINDEX = PEDINDEX + NSUBJECTS - COL + 1
              ENDIF
  30       continue  
        ELSE
  	   PEDINDEX = 1
  	   do 50 COL = 1,NSUBJECTS
              IF (iachar(BUFFER(COL,SNPCOL)) .NE. 3) THEN
  !		    PEDINDEX = 1 + ((COL-1)*MAPSIZE) - 
  !   /              ((COL-1)*((COL-1)-1)/2)
     		    
     		 do 60 ROW = COL,NSUBJECTS
     		    IF (iachar(BUFFER(ROW,SNPCOL)) .NE. 3)
     / 		    THEN
     		       SNPCOUNT(PEDINDEX) = SNPCOUNT(PEDINDEX) + 1
     		    ENDIF
     		    PEDINDEX = PEDINDEX + 1
  60   		 continue
  	      ELSE
  		 PEDINDEX = PEDINDEX + NSUBJECTS - COL + 1
              ENDIF
  50       continue                   
        ENDIF
  10  continue
      RETURN
      END 

      SUBROUTINE CORRPEDCALCWRAPPERTWO(VARSUM,EPED,FREQS,BUFFER,
     / NSUBJECTS,ARRAYSIZE,MAXSNPCOLS)
      implicit none
      INTEGER NSUBJECTS,ARRAYSIZE,MAXSNPCOLS,SNPCOLS
      REAL VARSUM(ARRAYSIZE)
      REAL EPED(ARRAYSIZE),FREQS(MAXSNPCOLS)
      CHARACTER BUFFER(NSUBJECTS,MAXSNPCOLS)
      SNPCOLS = 0
      CALL FILLBUFFERTHREAD(BUFFER,FREQS,SNPCOLS)
      DO WHILE ( SNPCOLS .NE. 0 )
      	CALL CORRPEDCALCTWO(BUFFER,EPED,FREQS,VARSUM,NSUBJECTS                 
     /       ,SNPCOLS,ARRAYSIZE)
      	CALL FILLBUFFERTHREAD(BUFFER,FREQS,SNPCOLS)
      END DO
      RETURN
      END
      
      SUBROUTINE CORRPEDCALCTWO(BUFFER,EPED,FREQS,VARSUM,NSUBJECTS,
     /SNPCOLS,ARRAYSIZE)
      implicit none
      INTEGER NSUBJECTS,ARRAYSIZE,SNPCOLS
      REAL VARSUM(ARRAYSIZE)
      REAL EPED(ARRAYSIZE),FREQS(SNPCOLS)
      CHARACTER BUFFER(NSUBJECTS,SNPCOLS)
      INTEGER SNPCOL
      INTEGER ROW,COL,ROWVAL,COLVAL,PEDINDEX
      REAL FREQ,VAR,NUMER
      
      do 10 SNPCOL = 1,SNPCOLS
	FREQ = FREQS(SNPCOL)
	IF (FREQ .EQ. -1.) GOTO 10
        VAR = (2.*FREQ*(1.-FREQ))
  	PEDINDEX = 1
  	do 30 COL = 1,NSUBJECTS
  	   COLVAL = iachar(BUFFER(COL,SNPCOL))
  	   IF (COLVAL .NE. 3) THEN
  !		    PEDINDEX = 1 + ((COL-1)*MAPSIZE) - 
  !   /              ((COL-1)*((COL-1)-1)/2)
     	     NUMER = (COLVAL - 2.*FREQ)
             do 40 ROW = COL,NSUBJECTS
     		ROWVAL = iachar(BUFFER(ROW,SNPCOL))
     		IF (ROWVAL .NE. 3) THEN
     		   EPED(PEDINDEX) = EPED(PEDINDEX) +
     /             NUMER*(ROWVAL-2.*FREQ)
     		   VARSUM(PEDINDEX) = VARSUM(PEDINDEX) + VAR
     		ENDIF
     		PEDINDEX = PEDINDEX + 1
  40   	     continue
  	   ELSE
  	     PEDINDEX = PEDINDEX + NSUBJECTS - COL + 1
           ENDIF
  30    continue
  10  continue
      RETURN
      END       
                   
           
      SUBROUTINE CORRPEDCALCIDLISTWRAPPERONE(SNPCOUNT,EPED,FREQS
     / ,BUFFER,ALPHA,NSUBJECTS,ARRAYSIZE,MAXSNPCOLS)
      implicit none
      INTEGER NSUBJECTS,ARRAYSIZE,MAXSNPCOLS,SNPCOLS
      INTEGER SNPCOUNT(ARRAYSIZE)
      REAL ALPHA, EPED(ARRAYSIZE),FREQS(MAXSNPCOLS)
      CHARACTER BUFFER(NSUBJECTS,MAXSNPCOLS) 
      SNPCOLS = 0 
      CALL FILLBUFFERTHREAD(BUFFER, FREQS, SNPCOLS)
      DO WHILE ( SNPCOLS .NE. 0 )
      	CALL CORRPEDCALCIDLISTONE(BUFFER,EPED,FREQS,SNPCOUNT,
     /ALPHA,NSUBJECTS,ARRAYSIZE,SNPCOLS)
     	CALL FILLBUFFERTHREAD(BUFFER,FREQS,SNPCOLS)
      END DO
      RETURN 
      END       

      SUBROUTINE CORRPEDCALCIDLISTONE(BUFFER,EPED,FREQS,SNPCOUNT,
     /ALPHA,NSUBJECTS,ARRAYSIZE,SNPCOLS)
      implicit none
      INTEGER NSUBJECTS,ARRAYSIZE,SNPCOLS
      INTEGER SNPCOUNT(ARRAYSIZE)
      REAL ALPHA,EPED(ARRAYSIZE),FREQS(SNPCOLS)
      CHARACTER BUFFER(NSUBJECTS,SNPCOLS)
      INTEGER SNPCOL
      INTEGER ROW,COL,ROWVAL,COLVAL,PEDINDEX
      REAL FREQ,VAR,NUMER
      do 10 SNPCOL = 1,SNPCOLS
	FREQ = FREQS(SNPCOL)
	IF (FREQ .EQ. -1.) GOTO 10
  	
  	IF (FREQ .NE. 0.0 .AND. FREQ .NE. 1.0) THEN
  	   VAR = (2.*FREQ*(1.-FREQ))**ALPHA
  	   PEDINDEX = 1
  	   do 30 COL = 1,NSUBJECTS
  	      COLVAL = iachar(BUFFER(COL,SNPCOL))
  	      IF (COLVAL .NE. 3) THEN
  !		    PEDINDEX = 1 + ((COL-1)*MAPSIZE) - 
  !   /              ((COL-1)*((COL-1)-1)/2)
     		 NUMER = (COLVAL - 2.*FREQ)*VAR
     		 do 40 ROW = COL,NSUBJECTS
     		    ROWVAL = iachar(BUFFER(ROW,SNPCOL))
     		    IF (ROWVAL .NE. 3) THEN
     		       EPED(PEDINDEX) = EPED(PEDINDEX) +
     /                 NUMER*(ROWVAL-2.*FREQ)
     		       SNPCOUNT(PEDINDEX) = SNPCOUNT(PEDINDEX) + 1
     		    ENDIF
     		    PEDINDEX = PEDINDEX + 1
  40   		 continue
  	      ELSE
  		 PEDINDEX = PEDINDEX + NSUBJECTS - COL + 1
              ENDIF
  30       continue  
        ELSE
  	   PEDINDEX = 1
  	   do 50 COL = 1,NSUBJECTS
              IF (iachar(BUFFER(COL,SNPCOL)) .NE. 3) THEN
  !		    PEDINDEX = 1 + ((COL-1)*MAPSIZE) - 
  !   /              ((COL-1)*((COL-1)-1)/2)
     		    
     		 do 60 ROW = COL,NSUBJECTS
     		    IF (iachar(BUFFER(ROW,SNPCOL)) .NE. 3)
     / 		    THEN
     		       SNPCOUNT(PEDINDEX) = SNPCOUNT(PEDINDEX) + 1
     		    ENDIF
     		    PEDINDEX = PEDINDEX + 1
  60   		 continue
  	      ELSE
  		 PEDINDEX = PEDINDEX + NSUBJECTS - COL + 1
              ENDIF
  50       continue                   
        ENDIF
  10  continue
      RETURN
      END
      
      
      SUBROUTINE CORRPEDCALCIDLISTWRAPPERTWO(VARSUM,EPED,FREQS
     / ,BUFFER,NSUBJECTS,ARRAYSIZE,MAXSNPCOLS)
      implicit none
      INTEGER NSUBJECTS,ARRAYSIZE,MAXSNPCOLS,SNPCOLS
      REAL VARSUM(ARRAYSIZE)
      REAL EPED(ARRAYSIZE),FREQS(MAXSNPCOLS)
      CHARACTER BUFFER(NSUBJECTS,MAXSNPCOLS) 
      SNPCOLS = 0 
      CALL FILLBUFFERTHREAD(BUFFER,FREQS,SNPCOLS)
      DO WHILE ( SNPCOLS .NE. 0 )
      	CALL CORRPEDCALCIDLISTTWO(BUFFER,EPED,FREQS,VARSUM,
     /NSUBJECTS,ARRAYSIZE,SNPCOLS)
     	CALL FILLBUFFERTHREAD(BUFFER,FREQS,SNPCOLS)
      END DO
      RETURN 
      END       

      SUBROUTINE CORRPEDCALCIDLISTTWO(BUFFER,EPED,FREQS,VARSUM,
     /NSUBJECTS,ARRAYSIZE,SNPCOLS)
      implicit none
      INTEGER NSUBJECTS,ARRAYSIZE,SNPCOLS
      REAL VARSUM(ARRAYSIZE)
      REAL EPED(ARRAYSIZE),FREQS(SNPCOLS)
      CHARACTER BUFFER(NSUBJECTS,SNPCOLS)
      INTEGER SNPCOL
      INTEGER ROW,COL,ROWVAL,COLVAL,PEDINDEX
      REAL FREQ,VAR,NUMER
      do 10 SNPCOL = 1,SNPCOLS
	FREQ = FREQS(SNPCOL)
  	IF (FREQ .EQ. -1.) GOTO 10
  	VAR = (2.*FREQ*(1.-FREQ))
  	PEDINDEX = 1
  	do 30 COL = 1,NSUBJECTS
  	   COLVAL = iachar(BUFFER(COL,SNPCOL))
  	   IF (COLVAL .NE. 3) THEN
  !		    PEDINDEX = 1 + ((COL-1)*MAPSIZE) - 
  !   /              ((COL-1)*((COL-1)-1)/2)
     	      NUMER = (COLVAL - 2.*FREQ)
     	      do 40 ROW = COL,NSUBJECTS
     		 ROWVAL = iachar(BUFFER(ROW,SNPCOL))
     		 IF (ROWVAL .NE. 3) THEN
     		    EPED(PEDINDEX) = EPED(PEDINDEX) +
     /              NUMER*(ROWVAL-2.*FREQ)
     		    VARSUM(PEDINDEX) = VARSUM(PEDINDEX) + VAR
     		 ENDIF
     		 PEDINDEX = PEDINDEX + 1
  40   	      continue
  	   ELSE
  	      PEDINDEX = PEDINDEX + NSUBJECTS - COL + 1
           ENDIF
  30    continue  
  10  continue
      RETURN
      END      
      
      SUBROUTINE KINGPEDCALCWRAPPER(BUFFER,SQRDDIFFSUM,HETEROCOL
     /,HETEROROW,NSUBJECTS,ARRAYSIZE,MAXSNPCOLS)
      implicit none
      INTEGER NSUBJECTS,ARRAYSIZE,MAXSNPCOLS,SNPCOLS
      INTEGER SQRDDIFFSUM(ARRAYSIZE)
      INTEGER HETEROCOL(ARRAYSIZE), HETEROROW(ARRAYSIZE)
      CHARACTER BUFFER(NSUBJECTS,MAXSNPCOLS)
  !    REAL VARSUM(ARRAYSIZE)
      SNPCOLS = 0
      CALL FILLBUFFERTHREADKING(BUFFER,SNPCOLS)
      DO WHILE ( SNPCOLS .NE. 0 )
  !    	CALL KINGPEDCALC(BUFFER,SQRDDIFFSUM,VARSUM,NSUBJECTS                 
  !   /       ,SNPCOLS,ARRAYSIZE)
  	CALL KINGPEDCALC(BUFFER, SQRDDIFFSUM, HETEROCOL,
     /  HETEROROW,NSUBJECTS, SNPCOLS, ARRAYSIZE)
      	CALL FILLBUFFERTHREADKING(BUFFER, SNPCOLS)
      END DO
      RETURN
      END 

      SUBROUTINE KINGPEDCALC(BUFFER, SQRDDIFFSUM, HETEROCOL,
     /  HETEROROW,NSUBJECTS, SNPCOLS, ARRAYSIZE)
      implicit none
      INTEGER NSUBJECTS,ARRAYSIZE,SNPCOLS,COL,ROW,COLVAL,ROWVAL,SNPCOL
      INTEGER SQRDDIFFSUM(ARRAYSIZE),MISSINGSUBJECTS,FREQSUM,PEDINDEX
      INTEGER HETEROCOL(ARRAYSIZE)
      INTEGER HETEROROW(ARRAYSIZE)
      CHARACTER BUFFER(NSUBJECTS,SNPCOLS)
      do 10 SNPCOL = 1,SNPCOLS
  	PEDINDEX = 1      
  	do 30 COL = 1,NSUBJECTS
  	   COLVAL = iachar(BUFFER(COL,SNPCOL))
  	   IF (COLVAL .NE. 3) THEN
     	      do 40 ROW = COL,NSUBJECTS
     		 ROWVAL = iachar(BUFFER(ROW,SNPCOL))
     		 IF (ROWVAL .NE. 3) THEN
     		   
     		    SQRDDIFFSUM(PEDINDEX) = SQRDDIFFSUM(PEDINDEX)
     /		     + (ROWVAL - COLVAL)**2
     		    IF (ROWVAL .EQ. 1) THEN
     		    	HETEROROW(PEDINDEX) = HETEROROW(PEDINDEX) + 1
     		    ENDIF
     		    IF(COLVAL .EQ. 1) THEN
     		    	HETEROCOL(PEDINDEX) = HETEROCOL(PEDINDEX) + 1
     		    ENDIF
     		 ENDIF
     		 PEDINDEX = PEDINDEX + 1
  40   	      continue
  	   ELSE
  	      PEDINDEX = PEDINDEX + NSUBJECTS - COL + 1
           ENDIF
  30    continue
  10  continue  
      RETURN
      END             

      SUBROUTINE KINGPEDCALCIDLISTWRAPPER(BUFFER,SQRDDIFFSUM,
     /HETEROCOL,HETEROROW,NSUBJECTS,ARRAYSIZE,MAXSNPCOLS)  
      implicit none
      INTEGER NSUBJECTS,ARRAYSIZE,SNPCOLS,MAXSNPCOLS
      INTEGER SQRDDIFFSUM(ARRAYSIZE),HETEROCOL(ARRAYSIZE)
      INTEGER HETEROROW(ARRAYSIZE)
      CHARACTER BUFFER(NSUBJECTS,MAXSNPCOLS)
      SNPCOLS = 0
      CALL FILLBUFFERTHREADKING(BUFFER,SNPCOLS)
      DO WHILE ( SNPCOLS .NE. 0 )
        CALL KINGPEDCALCIDLIST(BUFFER,SQRDDIFFSUM,HETEROCOL,
     /   HETEROROW,NSUBJECTS,ARRAYSIZE,SNPCOLS)
        CALL FILLBUFFERTHREADKING(BUFFER,SNPCOLS)
      END DO
      RETURN 
      END
      
      SUBROUTINE KINGPEDCALCIDLIST(BUFFER,SQRDDIFFSUM,HETEROCOL,
     /  HETEROROW,NSUBJECTS,ARRAYSIZE,SNPCOLS)
      implicit none
      INTEGER NSUBJECTS,ARRAYSIZE,SNPCOLS
      INTEGER SQRDDIFFSUM(ARRAYSIZE),HETEROCOL(ARRAYSIZE)
      INTEGER HETEROROW(ARRAYSIZE)
      CHARACTER BUFFER(NSUBJECTS,SNPCOLS)
      INTEGER SNPCOL,PINDEX,ROW,COL,ROWVALUE,COLVALUE  
        
      do 10 SNPCOL = 1,SNPCOLS
         PINDEX = 1  
         do 20 COL = 1,NSUBJECTS
            COLVALUE = iachar(BUFFER(COL,SNPCOL))
            IF ( COLVALUE .NE. 3 ) THEN
               do 30 ROW = COL,NSUBJECTS
                  ROWVALUE = iachar(BUFFER(ROW,SNPCOL))
                  IF(ROWVALUE .NE. 3) THEN
                     SQRDDIFFSUM(PINDEX) = SQRDDIFFSUM(PINDEX)
     /                 + (ROWVALUE-COLVALUE)**2
     		     IF(ROWVALUE .EQ. 1) THEN
     		        HETEROROW(PINDEX) = HETEROROW(PINDEX) + 1
     		     ENDIF
     		     IF(COLVALUE .EQ. 1) THEN
     		        HETEROCOL(PINDEX) = HETEROCOL(PINDEX) + 1
     		     ENDIF
     		  ENDIF
     		  PINDEX = PINDEX + 1
  30           continue
  	    ELSE
  	       PINDEX = PINDEX + NSUBJECTS - COL + 1  		                       
            ENDIF
  20     continue
  
  10  continue
      RETURN
      END        

	
      
      SUBROUTINE KINGPEDHOMOCALCWRAPPER(BUFFER,SQRDDIFFSUM,
     /VARSUM,NSUBJECTS,ARRAYSIZE,MAXSNPCOLS)
      
      implicit none
      INTEGER NSUBJECTS,ARRAYSIZE,SNPCOLS,MAXSNPCOLS
      INTEGER SQRDDIFFSUM(ARRAYSIZE)
      INTEGER HETEROROW(ARRAYSIZE),BUFFERROWS
      CHARACTER BUFFER(NSUBJECTS,MAXSNPCOLS)
      REAL VARSUM(ARRAYSIZE)
     
      SNPCOLS = 0
      CALL FILLBUFFERTHREADKING(BUFFER,SNPCOLS)
      DO WHILE ( SNPCOLS .NE. 0 )
        CALL KINGPEDHOMOCALC(BUFFER,SQRDDIFFSUM,VARSUM,
     /   NSUBJECTS,SNPCOLS,ARRAYSIZE)
        CALL FILLBUFFERTHREADKING(BUFFER,SNPCOLS)
      END DO
      RETURN 
      END
                  
      SUBROUTINE KINGPEDHOMOCALC(BUFFER,SQRDDIFFSUM,VARSUM,NSUBJECTS                 
     /       ,SNPCOLS,ARRAYSIZE)
      implicit none
      INTEGER NSUBJECTS,ARRAYSIZE,SNPCOLS,COL,ROW,COLVAL,ROWVAL,SNPCOL
      INTEGER SQRDDIFFSUM(ARRAYSIZE),MISSINGSUBJECTS,FREQSUM,PEDINDEX
      REAL VARSUM(ARRAYSIZE),FREQ,VAR
      CHARACTER BUFFER(NSUBJECTS,SNPCOLS)

      do 10 SNPCOL = 1,SNPCOLS
      	FREQSUM = 0
      	MISSINGSUBJECTS = 0
      	do 20 ROW = 1,NSUBJECTS
      		ROWVAL = iachar(BUFFER(ROW,SNPCOL))
      		IF (ROWVAL .NE. 3) THEN
      			FREQSUM = FREQSUM + ROWVAL
      		ELSE
      			MISSINGSUBJECTS=MISSINGSUBJECTS+1
      		ENDIF
  20	continue
  	IF (MISSINGSUBJECTS .EQ. NSUBJECTS) goto 10
  	
  	FREQ = FREQSUM
  	FREQ = FREQ/(2.*(NSUBJECTS-MISSINGSUBJECTS))
  	IF (FREQ .EQ. 1. .OR. FREQ .EQ. 0.) goto 10

        VAR = (2.*FREQ*(1.-FREQ))
  	PEDINDEX = 1      
  	do 30 COL = 1,NSUBJECTS
  	   COLVAL = iachar(BUFFER(COL,SNPCOL))
  	   IF (COLVAL .NE. 3) THEN
  !		    PEDINDEX = 1 + ((COL-1)*MAPSIZE) - 
  !   /              ((COL-1)*((COL-1)-1)/2)
     	      do 40 ROW = COL,NSUBJECTS
     		 ROWVAL = iachar(BUFFER(ROW,SNPCOL))
     		 IF (ROWVAL .NE. 3) THEN
     		    
     		    SQRDDIFFSUM(PEDINDEX) = SQRDDIFFSUM(PEDINDEX)
     /		     + (ROWVAL - COLVAL)**2
     		    VARSUM(PEDINDEX) = VARSUM(PEDINDEX) + VAR
     		 ENDIF
     		 PEDINDEX = PEDINDEX + 1
  40   	      continue
  	   ELSE
  	      PEDINDEX = PEDINDEX + NSUBJECTS - COL + 1
           ENDIF
  30    continue
  10  continue  
      RETURN
      END      

