	PROGRAM DISPERSION
	IMPLICIT NONE

	REAL :: PSTAG,PDOWN,AU,AV,AP,UIN,VIN,VISA,ROA,VISB,ROB,daby
	INTEGER :: I,J,NI,NJ,it,vtk
	REAL,DIMENSION(8000,1000) :: X,Y,AREA,ROO,VX,VY,P,VISC,
     &  X1,Y1,FACE_U,FACE_V,MASS,DAB, STRF,VORT
	REAL,DIMENSION(8000) :: DX,DY
	CHARACTER(LEN=10) :: NOME

	COMMON NI,NJ,PDOWN,X,Y,AREA,ROO,VX,VY,P,VISC,X1,Y1,DX,DY,
     &  AU,AV,AP,UIN,VIN,it,FACE_U,FACE_V,MASS,DAB,daby,strf,vort

	OPEN(UNIT=2,FILE='flow1')

C	LENDO OS VALORES DE VISCOSIDADE, MASSA ESPECIFICA, PRESSAO
C       DE ESTAGNACAO E PRESSAO NA SAIDA

	READ(2,*) NI,NJ
	READ(2,*) VISA,ROA,VISB,ROB
	READ(2,*) DABy
	READ(2,*) UIN,VIN,PDOWN
	READ(2,*) AU,AV,AP
	READ(2,*) it,vtk	
C
        DO 1 I=1,NI+2
		DO 2 J=1,NJ+2
		X(I,J)=0.
		X1(I,J)=0.
		Y(I,J)=0.
		Y1(I,J)=0.
		AREA(I,J)=0.
		ROO(I,J)=0.
		VX(I,J)=0.
		VY(I,J)=0.
		P(I,J)=0.
		VISC(I,J)=0.
		FACE_V(I,J)=0.
		FACE_U(I,J)=0.
		MASS(I,J)=0.
		DAB(I,J)=0.
		STRF(I,J)=0.
		VORT(I,J)=0.
2		CONTINUE
	DX(I)=0.
	DY(I)=0.
1	CONTINUE

      	CALL GRID(NI,NJ,NOME,X1,Y1,AREA,DX,DY)	
	CALL SECONDARY_GRID(NI,NJ,VX,VY,X1,Y1,X,Y)
	CALL INGUESS(NI,NJ,VISC,VX,VY,ROO,P,FACE_U,FACE_V,MASS,
     &  DAB,daby,VISA,ROA,VISB,ROB)
C	CALL BOUND_COND(NI,NJ,VX,VY,PDOWN,P,UIN,VIN,ROO,FACE_U,
C     &  FACE_V,MASS,VISC,VISA,VISB,ROA,ROB)
	CALL SIMPLE(NI,NJ,ROO,VX,VY,P,VISC,X1,Y1,X,Y,AU,AV,AP,it,UIN,
     &  VIN,FACE_U,FACE_V,MASS,DAB,vtk,VISA,VISB,ROA,ROB)
     	CALL VORTICITY(NI,NJ,VX,VY,X,Y,X1,Y1,VORT,it)
     	CALL STREAMFUNCTION(NI,NJ,VX,VY,X,Y,X1,Y1,STRF,VORT)
	CALL CONVERTING(NI,NJ,P,X1,Y1,X,Y,ROO,VX,VY,STRF)
	CALL PLOTTING(NI,NJ,X1,Y1,ROO,VISC,VX,VY,P,X,Y,MASS,STRF,VORT)

	END PROGRAM DISPERSION

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

	SUBROUTINE GRID(NI,NJ,TITLE,X1,Y1,AREA,DX,DY)

        Parameter (i_max=8000,j_max=1000)
	INTEGER :: I,J,reset
	REAL :: SUM1,sum2,H
	REAL,DIMENSION(i_max,j_max) :: X1,Y1,AREA,NORM_I,NORM_J,SUMM
	REAL,DIMENSION(i_max) :: XLOW,YLOW,XHIGH,YHIGH,DX,DY
	CHARACTER(LEN=10) :: TITLE

	OPEN(UNIT=1,FILE='geom')

C       ABERTURA DO ARQUIVO "GEOM"
	
	READ(1,*)  TITLE

C       LEITURA DOS DADOS NO ARQUIVO "GEOM"

	DO 102 I=1,NI+1
	READ(1,*) XLOW(I),YLOW(I),XHIGH(I),YHIGH(I)
102     CONTINUE


C
C
C       CALCULO DOS GAPS NAS DUAS DIRECOES
	DO 103 I=1,NI+1
	DX(I)=(XHIGH(I)-XLOW(I))/(NJ)
	DY(I)=(YHIGH(I)-YLOW(I))/(NJ)
	X1(I,1)=XLOW(I)
	Y1(I,1)=YLOW(I)
		DO 104 J=2,NJ+1
		X1(I,J)=X1(I,J-1)+DX(I)
		Y1(I,J)=Y1(I,J-1)+DY(I)
104		CONTINUE
103	CONTINUE



C
C       CALCULO DAS AREAS DE CADA CELULA

	DO 105 I=1,NI
	      DO 106 J=1,NJ
	      AREA(I,J)=0.5*(((X1(I+1,J)-X1(I,J))*(Y1(I+1,J+1)-Y1(I+1,J)))
     &	      +((X1(I+1,J+1)-X1(I,J+1))*(Y1(I,J+1)-Y1(I,J))))

C
C
C        VERIFICACAO DAS AREAS DAS CELULAS

		IF (AREA(I,J)<0) THEN
		PRINT*,"AREA NEGATIVA"
		EXIT
		ELSE
			IF (AREA(I,J)==0.) THEN
			PRINT*,"AREA NULA"
			EXIT
			END IF
		END IF
c
c
c           NORMALIZAÇÃO DO VETOR PERPENDICULAR A CADA FACE

	      NORM_I(I,J)=SQRT(((X1(I,J+1)-X1(I,J))**2)+((Y1(I,J)
     &        -Y1(I,J+1))**2))

	      NORM_J(I,J)=SQRT(((X1(I+1,J)-X1(I,J))**2)+((Y1(I,J)
     &        -Y1(I+1,J))**2))

c		calculating the representative grid size h

		SUM2=SUM2+AREA(i,j)


106	      CONTINUE
105	CONTINUE

	
	H=SQRT(SUM2/(NI*NJ))

	PRINT*,'REPRESENTATIVE GRID SIZE H:',H

C      VERIFICACAO SE OS VETORES NORMAIS EM CADA FACE SE CANCELAM
	reset=0
	DO 107 I=1,NI
		DO 108 J=1,NJ

	         SUMM(I,J)=NORM_I(I+1,J)-NORM_I(I,J)+NORM_J(I,J+1)-
     &           NORM_J(I,J)

	      IF (ABS(SUMM(I,J))>1.E-6) reset=1
108     	CONTINUE
107	CONTINUE

c       CONCLUSAO DA VERIFICACAO
	IF (abs(SUMM(I,J))<=1.E-6) THEN
		PRINT*,'Somatorio entre vetores esta ok'
	ELSE
		PRINT*, "Somatorio nao esta adequado"
	ENDIF

	CLOSE(1)

	RETURN	
	END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

	SUBROUTINE SECONDARY_GRID(NI,NJ,VX,VY,X1,Y1,X,Y)

        Parameter (i_max=8000,j_max=1000)
	INTEGER :: I,J
	REAL,DIMENSION(i_max,j_max) :: RE,VX,VY,X,Y,X1,Y1

	DO I=1,NI+2
		DO J=1,NJ+2
		IF (I==1) THEN
		X(I,J)=X1(I,J)
		ELSE
			IF (I<=NI+1) THEN
			X(I,J)=X1(I-1,J)+(0.5*(X1(I,J)-X1(I-1,J)))
			ELSE
			X(I,J)=X1(I-1,J)
			END IF
		END IF

		IF (J==NJ+2) X(I,J)=X(I,J-1)

		END DO
	END DO


	DO I=1,NI+2
		DO J=1,NJ+2
		IF (J==1) THEN
		Y(I,J)=Y1(I,J)
		ELSE
			IF (J<=NJ+1) THEN
			Y(I,J)=Y1(I,J-1)+(0.5*(Y1(I,J)-Y1(I,J-1)))
			ELSE
			Y(I,J)=Y1(I,J-1)
			END IF
		END IF

		IF (I==NI+2) Y(I,J)=Y(I-1,J)
		END DO
	END DO

	RETURN
	END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

	SUBROUTINE INGUESS(NI,NJ,VISC,VX,VY,ROO,P,FACE_U,FACE_V,MASS,
     &  DAB,daby,VISA,ROA,VISB,ROB)

        Parameter (i_max=8000,j_max=1000)
	REAL,DIMENSION(i_max,j_max) :: ROO,VX,VY,P,VISC,FACE_U,FACE_V,
     &  MASS,DAB

C       ATRIBUINDO OS VALORES CONSTANTES DE VELOCIDADE, E O CHUTE
C       INICIAL PARA O CAMPO DE PRESSAO E DE VELOCIDADE

	DO I=1,NI+2
		DO J=1,NJ+2

		VX(I,J)=0.
c
		VY(I,J)=0.

		ROO(I,J)=ROB

		VISC(I,J)=VISB
c
		P(I,J)=0.0005

		FACE_U(I,J)=0.

		FACE_V(I,J)=0.
		
		MASS(I,J)=0.0

		DAB(I,J)=DABy
c
		END DO
	END DO


	RETURN
	END


CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

	SUBROUTINE BOUND_COND(NI,NJ,VX,VY,PDOWN,P,UIN,VIN,ROO,FACE_U,
     &  FACE_V,MASS,VISC,VISA,VISB,ROA,ROB)
	
        Parameter (i_max=8000,j_max=1000)
	INTEGER :: I,J
	REAL :: VISA,VISB,ROA,ROB,UIN,VIN
	REAL,DIMENSION(i_max,j_max) :: VX,VY,P,ROO,FACE_U,FACE_V,MASS,
     &  VISC


C	NO SLIP CONDITION

	DO I=1,NI+2

C	P(I,1)=PDOWN

c	NORTH AND SOUTH WALLS

	VX(I,NJ+2)=0.
	VY(I,NJ+2)=0.
	FACE_V(I,NJ+1)=0.
	VY(I,1)=0.
	VX(I,1)=0.
	FACE_V(I,1)=0.
	MASS(I,1)=MASS(I,2)
	ROO(I,NJ+2)=ROO(I,NJ+1)
	VISC(I,NJ+2)=VISC(I,NJ+1)
	MASS(I,NJ+2)=MASS(I,NJ+1)
	ROO(I,1)=ROO(I,2)
	VISC(I,1)=VISC(I,2)


	END DO

c	west boundary

	DO J=1,nj+2

	VX(1,J)=VIN
	VY(1,J)=0.
	FACE_U(1,J)=VIN
	MASS(1,J)=0.0
	VISC(1,J)=VISB
	ROO(1,J)=ROB

c	outlet condition / east boundary

	P(NI+2,J)=PDOWN
	MASS(NI+2,J)=MASS(NI+1,J)
	VISC(NI+2,J)=VISC(NI+1,J)
	ROO(NI+2,J)=ROO(NI+1,J)

	END DO

	
	VY(12,1)=UIN
	VX(12,1)=0.0
	FACE_V(12,1)=UIN
	MASS(12,1)=1.0
	VISC(12,1)=VISA
	ROO(12,1)=ROA

	

	MASS(NI+2,NJ+2)=MASS(NI+2,NJ+1)
	MASS(NI+2,1)=MASS(NI+2,2)

	RETURN
	END


CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

	SUBROUTINE MASS_FRACTION(NI,NJ,X1,Y1,X,Y,VX,VY,ROO,VISC,FACE_U,
     &  FACE_V,FACEJ_U,MASS,DAB)

        Parameter (i_max=8000,j_max=1000)
	INTEGER :: I,J,NI,NJ,NeqMASS,g,m,MODE
	REAL,DIMENSION(i_max,j_max) :: ROO,VX,VY,VISC,X,Y,X1,Y1,AEM,
     &  AWM,FACE_U,FACE_V,FACEJ_U,ANM,ASM,SSM,APM,DWM,FWM,DEM,MASS,DAB,
     &	FEM,DNM,FNM,DSM,FSM
	REAL :: MASSrec(i_max)

	DO J=2,NJ+1

		DO I=2,NI+1

C		WEST BOUNDARY

		IF (I==2) THEN

		DWM(2,J)=0.
c
		FWM(2,J)=0.
c
		ELSE

		DWM(I,J)=DAB(I,J)*(Y1(I-1,J)-Y1(I-1,J-1))/
     &		(0.5*(X(I,J)-X(I-1,J)+X(I,J-1)-X(I-1,J-1)))

		FWM(I,J)=(FACE_U(I-1,J)*(Y1(I-1,J)-Y1(I-1,J-1)))

		END IF

C		EAST BOUNDARY

		IF (I==NI+1) THEN

		DEM(I,J)=0.

		FEM(NI+1,J)=0.

		ELSE

		DEM(I,J)=(DAB(I,J)*(Y1(I,J)-Y1(I,J-1))/
     &		(0.5*(X(I+1,J)-X(I,J)+X(I+1,J-1)-X(I,J-1))))

		FEM(I,J)=(FACE_U(I,J)*(Y1(I,J)-Y1(I,J-1)))

		END IF

C		SOUTH BOUNDARY

		IF (J==2) THEN

		DSM(I,2)=0.

		FSM(I,2)=0.

		ELSE

		DSM(I,J)=(DAB(I,J)*(X1(I,J-1)-X1(I-1,J-1))/
     &		(0.5*(Y(I,J)-Y(I,J-1)+Y(I-1,J)-Y(I-1,J-1))))

                FSM(I,J)=(FACE_V(I,J-1)*(X1(I,J-1)-X1(I-1,J-1)))

		FSM(I,J)=FSM(I,J)+(FACEJ_U(I,J-1)*(Y1(I,J-1)-
     &          Y1(I-1,J-1)))

		END IF

C		NORTH BOUNDARY

		IF (J==NJ+1) THEN

		DNM(I,NJ+1)=0.

		FNM(I,NJ+1)=0.

		ELSE

		DNM(I,J)=(DAB(I,J)*(X1(I,J)-X1(I-1,J))/
     &		(0.5*(Y(I,J+1)-Y(I,J)+Y(I-1,J+1)-Y(I-1,J))))

                FNM(I,J)=(FACE_V(I,J)*(X1(I,J)-X1(I-1,J)))

		FNM(I,J)=FNM(I,J)+(FACEJ_U(I,J)*(Y1(I,J)-Y1(I-1,J)))

		END IF

		SSM(I,J)=0.

C              	HYBRID SCHEME

C	Determining the link coefficients of the momentum equation

		AWM(I,J)=AMAX1(DWM(I,J)+AMAX1(FWM(I,J),0.0),DWM(I,J)+
     &				(FWM(I,J)/2))

		AEM(I,J)=AMAX1(DEM(I,J)+AMAX1(-FEM(I,J),0.0),DEM(I,J)-
     &				(FEM(I,J)/2))

		ASM(I,J)=AMAX1(DSM(I,J)+AMAX1(FSM(I,J),0.0),DSM(I,J)+
     &				(FSM(I,J)/2))

		ANM(I,J)=AMAX1(DNM(I,J)+AMAX1(-FNM(I,J),0.0),DNM(I,J)-
     &				(FNM(I,J)/2))

		APM(I,J)=AWM(I,J)+AEM(I,J)+ASM(I,J)+ANM(I,J)+
     &		        FEM(I,J)-FWM(I,J)+FNM(I,J)-FSM(I,J)

		END DO

        END DO

	DO J=2,NJ+1

		DO I=2,NI+1

C		BOUNDARY CONDITIONS

C               MAIN INLET

		IF (I==2) THEN

c		MASS(1,J)=1.0

                APM(2,J)=APM(2,J)+(DAB(I,J)*((0.5*(Y1(2,J)-Y1(2,J-1)+
     &                   Y1(1,J)-Y1(1,J-1)))/(3*0.5*(X1(2,J)-X1(1,J)
     &                   +X1(2,J-1)-X1(1,J-1)))))

                AEM(2,J)=AEM(2,J)+(DAB(I,J)*(Y1(2,J)-Y1(2,J-1))/
     &                  (3*(0.5*(X1(2,J)-X1(1,J)+X1(2,J-1)-X1(1,J-1))))))

                SSM(2,J)=SSM(2,J)+(VX(1,J)*MASS(1,j)*(0.5*(Y1(2,J)-
     &		        Y1(2,J-1)+Y1(1,J)-Y1(1,J-1))))+(DAB(I,J)*
     &			MASS(1,J)*(8*(0.5*(Y1(2,J)-Y1(2,J-1)+Y1(1,J)-
     &                  Y1(1,J-1)))/(3*(0.5*(X1(2,J)-X1(1,J)+X1(2,J-1)-
     &			X1(1,J-1))))))

		ELSE

C               OUTLET

		  IF (I==NI+1) THEN

                  APM(NI+1,J)=APM(NI+1,J)+(DAB(I,J)*((0.5*(Y1(I,J)-
     &		   Y1(I,J-1)+Y1(I-1,J)-Y1(I-1,J-1)))/(3*0.5*(X1(I,J)-
     &             X1(I-1,J)+X1(I,J-1)-X1(I-1,J-1)))))+(VX(NI+1,J)
     &             *(0.5*(Y1(I,J)-Y1(I,J-1)+Y1(I-1,J)-Y1(I-1,J-1))))

                  AWM(NI+1,J)=AWM(NI+1,J)+(DAB(I,J)*(Y1(I,J)-
     &              Y1(I,J-1))/(3*(0.5*(X1(I,J)-X1(I-1,J)+X1(I,J-1)-
     &             X1(I-1,J-1)))))

c		  MASS(NI+2,J)=MASS(NI+1,J)


		  END IF

		END IF


C               SOUTH WALL

		IF (J==2) THEN

		   IF (I<12 .or. I>12) THEN

                   APM(I,2)=APM(I,2)+(DAB(I,J)*(0.5*(X1(I,J)-X1(I-1,J)
     &             +X1(I,J-1)-X1(I-1,J-1)))/(3*0.5*(Y1(I,J)-Y1(I,J-1)
     &		   +Y1(I-1,J)-Y1(I-1,J-1))))

                   ANM(I,2)=ANM(I,2)+(DAB(I,J)*((X1(I,J-1)-X1(I-1,J-1))/
     &             (3*0.5*(Y1(I,J)-Y1(I,J-1)+Y1(I-1,J)-Y1(I-1,J-1)))))

c		   MASS(I,1)=MASS(I,2)

		   ELSE

C               SIDE INLET

			IF (I==12) THEN

c		   MASS(I,1)=0.0

                   APM(I,2)=APM(I,2)+(DAB(I,J)*(3*(0.5*(X1(I,J)-
     &             X1(I-1,J)+X1(I,J-1)-X1(I-1,J-1))))/(0.5*(Y1(I,J)
     &		   -Y1(I,J-1)+Y1(I-1,J)-Y1(I-1,J-1))))

                   ANM(I,2)=ANM(I,2)+(DAB(I,J)*(X1(I,J)-X1(I-1,J))/
     &              (3*(0.5*(Y1(I,J)-Y1(I,J-1)+Y1(I-1,J)-Y1(I-1,J-1)))))

                   SSM(I,2)=SSM(I,2)+(VY(I,1)*MASS(I,1)*(X1(I,J-1)-
     &              X1(I-1,J-1)))+(DAB(I,J)*MASS(I,1)*(8*(X1(I,J-1)-
     &              X1(I-1,J-1)))/(3*(0.5*(Y1(I,J)-Y1(I,J-1)+Y1(I-1,J)-
     & 		    Y1(I-1,J-1)))))

		        END IF

		  END IF

		ELSE

C               NORTH WALL

		  IF (J==NJ+1) THEN

                  APM(I,NJ+1)=APM(I,NJ+1)+(DAB(I,J)*((0.5*(X1(I,J)-
     &             X1(I-1,J)+X1(I,J-1)-X1(I-1,J-1)))/(3*0.5*(Y1(I,J)-
     &             Y1(I,J-1)+Y1(I-1,J)-Y1(I-1,J-1)))))

                  ASM(I,NJ+1)=ASM(I,NJ+1)+(DAB(I,J)*(X1(I,J)-X1(I-1,J))/
     &             (3*0.5*(Y1(I,J)-Y1(I,J-1)+Y1(I-1,J)-Y1(I-1,J-1))))

c		  MASS(I,NJ+2)=MASS(I,NJ+1)


		  END IF

		END IF

		END DO

	END DO


		MODE=3

	NeqMASS=(NI+2)*(NJ+2)

c	DO it=1,50

C
	g=NeqMASS
C
	DO J=1,NJ+2
C
        g=g-(NI+2)
        m=0

 	      	DO I=1,NI+2

                m=m+1

		MASSrec(g+m)=MASS(I,J)

		END DO

	END DO

	CALL COEFF_TDMA(NeqMASS,NeqMASS,MASSrec,APM,
     &  AEM,AWM,ANM,ASM,SSM,NI,NJ,MODE)



        DO J=1,NJ+2

	  g=NeqMASS-(J*(NI+2))

	  m=0

		DO I=1,NI+2

		m=m+1

		MASS(I,J)=MASSrec(g+m)

		END DO

	  END DO


	RETURN
	END


CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

	SUBROUTINE SIMPLE(NI,NJ,ROO,VX,VY,P,VISC,X1,Y1,X,Y,ALFU,ALFV,
     &  ALFP,itt,UIN,VIN,FACE_U,FACE_V,MASS,DAB,vtk,VISA,VISB,ROA,
     &  ROB)

        Parameter (i_max=8000,j_max=1000)
	INTEGER :: I,J,NI,NJ,NeqP,g,m,it,Neq,NeqV,MODE,NeqMASS,vtk,
     &  vtk1
	REAL :: A,ALFU,ALFV,ALFP,c,d,n,s,w,e,limite,SSUM,UIN,VIN,
     &  VISA,VISB,ROA,ROB,Error_mass
	REAL,DIMENSION(i_max,j_max) :: ROO,VX,VY,P,VISC,X,Y,X1,Y1,
     &  DE,DW,DN,DS,FE,FW,FN,FS,AE,AW,AN,AS,AP,PU,PV,ERROR,VX_OLD,
     &  VY_OLD,AEp,AWp,ANp,ASp,APp,Pcor,Ppas,FACE_U,FACE_V,FACEJ_U,
     &  MASS_it,difference,MASS,DAB,STRF,VORT
	REAL,DIMENSION(i_max,j_max) :: AU_TDMA
	REAL,DIMENSION(i_max,j_max) :: AV_TDMA
	REAL,DIMENSION(i_max,j_max) :: AP_TDMA
	REAL :: Prec(i_max),VXrec(i_max),VYrec(i_max)

	vtk1=vtk

	DO I=1,NI+2
		DO J=1,NJ+2

		Pcor(I,J)=0.
		FACEJ_U(I,J)=0.
		STRF(I,J)=0.
		VORT(I,J)=0.

		END DO
	END DO


	OPEN(UNIT=13, FILE='error.dat')

	DO it=1,itt


	CALL BOUND_COND(NI,NJ,VX,VY,PDOWN,P,UIN,VIN,ROO,FACE_U,
     &  FACE_V,MASS,VISC,VISA,VISB,ROA,ROB)


C	Cálculo dos Fluxos Difusivos D e Fluxos Convectivos C
c	em cada face da célula (Capitulo 6 - Malalasekera) e em
c	cada componente de velocidade

C	Velocity components

	DO J=2,NJ+1

		DO I=2,NI+1

c		WEST BOUNDARY

		IF (I==2) THEN

		DW(2,J)=0.

		FW(2,J)=0.

		ELSE

		DW(I,J)=VISC(I,J)*(Y1(I-1,J)-Y1(I-1,J-1))/
     &		(0.5*(X(I,J)-X(I-1,J)+X(I,J-1)-X(I-1,J-1)))

		FW(I,J)=(ROO(I,J)*FACE_U(I-1,J)*(Y1(I-1,J)-
     &		Y1(I-1,J-1)))

		END IF


c		EAST BOUNDARY

		IF (I==NI+1) THEN

		DE(NI+1,J)=0.

		FE(NI+1,J)=0.

		ELSE

		DE(I,J)=(VISC(I,J)*(Y1(I,J)-Y1(I,J-1))/
     &		(0.5*(X(I+1,J)-X(I,J)+X(I+1,J-1)-X(I,J-1))))

		FE(I,J)=(ROO(I,J)*FACE_U(I,J)*(Y1(I,J)-
     &		Y1(I,J-1)))

		END IF


c		SOUTH BOUNDARY

		IF (J==2) THEN

		DS(I,2)=0.

                FS(I,2)=0.

		ELSE

		DS(I,J)=(VISC(I,J)*(X1(I,J-1)-X1(I-1,J-1))/
     &		(0.5*(Y(I,J)-Y(I,J-1)+Y(I-1,J)-Y(I-1,J-1))))
  
                FS(I,J)=(ROO(I,J)*FACE_V(I,J-1)*(X1(I,J-1)-
     &		X1(I-1,J-1)))

		FS(I,J)=FS(I,J)+(ROO(I,J)*FACEJ_U(I,J-1)*
     &		(Y1(I,J-1)-Y1(I-1,J-1)))

		END IF


C		NORTH BOUNDARY

		IF (J==NJ+1) THEN

		DN(I,NJ+1)=0.

                FN(I,NJ+1)=0.

		ELSE

		DN(I,J)=(VISC(I,J)*(X1(I,J)-X1(I-1,J))/
     &		(0.5*(Y(I,J+1)-Y(I,J)+Y(I-1,J+1)-Y(I-1,J))))

                FN(I,J)=(ROO(I,J)*FACE_V(I,J)*(X1(I,J)-
     &		X1(I-1,J)))

		FN(I,J)=FN(I,J)+(ROO(I,J)*FACEJ_U(I,J)*
     &		(Y1(I,J)-Y1(I-1,J)))

		END IF


C	HYBRID SCHEME

c	Determining the link coefficients of the momentum equation

		AW(I,J)=AMAX1(FW(I,J),DW(I,J)+
     &				(FW(I,J)/2),0.0)
	
		AE(I,J)=AMAX1(-FE(I,J),DE(I,J)-
     &				(FE(I,J)/2),0.0)

		AS(I,J)=AMAX1(FS(I,J),DS(I,J)+
     &				(FS(I,J)/2),0.0)

		AN(I,J)=AMAX1(-FN(I,J),DN(I,J)-
     &				(FN(I,J)/2),0.0)

		AP(I,J)=AW(I,J)+AE(I,J)+AS(I,J)+AN(I,J)+
     &		        FE(I,J)-FW(I,J)+FN(I,J)-FS(I,J)

		end do

	end do


	DO J=2,NJ+1

		DO I=2,NI+1


C		BOUNDARY CONDITIONS

C		X-MOMENTUM 

		IF (I==2) THEN

C		PRESSURE ON THE INLET

		P(1,J)=(1.5*P(2,J))-(0.5*P(3,J))

		PU(2,J)=((0.5*(Y1(I,J)-Y1(I,J-1)+Y1(I-1,J)
     &		-Y1(I-1,J-1)))*(P(1,J)-(0.5*(P(2,J)+P(3,J)))))

C		INLET - TAYLOR SERIES EXPANSION AT TWO POINTS

		AE(2,J)=AE(2,J)+(VISC(2,J)*(Y1(2,J)-Y1(2,J-1))
     &		/(3.*(0.5*(X1(I,J)-X1(I-1,J)+X1(I,J-1)-
     &		X1(I-1,J-1)))))

		AP(2,J)=AP(2,J)+(3.*VISC(2,J)*(0.5*(Y1(2,J)-
     &		Y1(2,J-1)+Y1(1,J)-Y1(1,J-1)))/(0.5*(X1(2,J)-
     &		X1(1,J)+X1(2,J-1)-X1(1,J-1))))

		PU(2,J)=PU(2,J)+((0.5*(Y1(2,J)-Y1(2,J-1)+
     &		Y1(1,J)-Y1(1,J-1)))*(ROO(1,J)*VX(1,J)*VX(1,J)))+
     &		((0.5*(Y1(2,J)-Y1(2,J-1)+Y1(1,J)-Y1(1,J-1)))*
     &		((8.*VISC(1,J)*VX(1,J))/(3.*(0.5*(X1(2,J)-X1(1,J)+
     &		X1(2,J-1)-X1(1,J-1))))))


		ELSE

C		  OUTLET

		  IF (I==NI+1) THEN

		  PU(NI+1,J)=((0.5*(Y1(I,J)-Y1(I,J-1)+Y1(I-1,J)
     &		  -Y1(I-1,J-1)))*((0.5*(P(NI+1,J)+P(NI,J)))
     &		  -P(NI+2,J)))

		  AP(NI+1,J)=AP(NI+1,J)+(1.5*(0.5*(Y1(I,J)-Y1(I,J-1)
     &		  +Y1(I-1,J)-Y1(I-1,J-1)))*ROO(NI+1,J)*VX(NI+1,J))

		  AW(NI+1,J)=AW(NI+1,J)+(0.5*(0.5*(Y1(I,J)-Y1(I,J-1)
     &		  +Y1(I-1,J)-Y1(I-1,J-1)))*ROO(NI+1,J)*VX(NI+1,J))

		  ELSE

		  PU(I,J)=(0.5*(0.5*(Y1(I,J)-Y1(I,J-1)+Y1(I-1,J)
     &		  -Y1(I-1,J-1)))*(P(I-1,J)-P(I+1,J)))

		  END IF

		END IF


C		Y-MOMENTUM

		IF (J==2) THEN

C		SOURCE TERM OF PRESSURE IN THE SOUTH BOUNDARY

		P(I,1)=(1.5*P(I,2))-(0.5*P(I,3))

		PV(I,2)=((0.5*(X1(I,2)-X1(I-1,2)+X1(I,1)-
     &		X1(I-1,1)))*(P(I,1)-(0.5*(P(I,2)+P(I,3)))))

		PU(I,2)=PU(I,2)+(0.5*(Y1(I,2)-Y1(I-1,2)+Y1(I,1)-
     &		Y1(I-1,1)))*(P(I,1)-(0.5*(P(I,2)+P(I,3))))

C		  BOUNDARY CONDITION FOR THE INLET

		  IF (I==12) THEN

		  PV(I,2)=PV(I,2)+((0.5*(X1(I,2)-X1(I-1,2)+
     &		  X1(I,1)-X1(I-1,1)))*(ROO(I,1)*VY(I,1)*VY(I,1)))+
     &		  ((0.5*(X1(I,2)-X1(I-1,2)+X1(I,1)-X1(I-1,1)))*
     &		  ((8.*VISC(I,1)*VY(I,1))/(3.*(0.5*(Y1(I,2)-Y1(I,1)+
     &		  Y1(I-1,2)-Y1(I-1,1))))))

		  END IF


C		SOUTH WALL - TAYLOR SERIES EXPANSION AT TWO POINTS

		AN(I,2)=AN(I,2)+(VISC(I,2)*(X1(I,J-1)-
     &		X1(I-1,J-1))/(3.*(0.5*(Y1(I,J)-Y1(I,J-1)+
     &		Y1(I-1,J)-Y1(I-1,J-1)))))

		AP(I,2)=AP(I,2)+((3.*VISC(I,2)*(X1(I,1)-
     &		X1(I-1,1)))/(0.5*(Y1(I,2)-Y1(I,1)+
     &		Y1(I-1,2)-Y1(I-1,1))))

		ELSE

		  IF (J==NJ+1) THEN

C		  SOURCE TERM OF PRESSURE FOR Y-MOMENTUM - NORTH

		  P(I,NJ+2)=1.5*P(I,NJ+1)-0.5*P(I,NJ)

		  PV(I,NJ+1)=((0.5*(X1(I,NJ+1)-X1(I-1,NJ+1)
     &		  +X1(I,NJ)-X1(I-1,NJ)))*((0.5*(P(I,NJ+1)+
     &		  P(I,NJ)))-P(I,NJ+2)))

		  PU(I,NJ+1)=PU(I,NJ+1)+((0.5*(Y1(I,NJ+1)-Y1(I-1,NJ+1)
     &		  +Y1(I,NJ)-Y1(I-1,NJ)))*((0.5*(P(I,NJ+1)+P(I,NJ)))-
     &		  P(I,NJ+2)))

C		  NORTH WALL - TAYLOR SERIES EXPANSION AT TWO POINTS

		  AS(I,NJ+1)=AS(I,NJ+1)+(VISC(I,NJ+1)*(X1(I,J)-
     &		  X1(I-1,J))/(3.*(0.5*(Y1(I,J)-Y1(I,J-1)+
     &		  Y1(I-1,J)-Y1(I-1,J-1)))))

		  AP(I,NJ+1)=AP(I,NJ+1)+(3.*VISC(I,NJ+1)*(X1(I,NJ+1)-
     &		  X1(I-1,NJ+1))/(0.5*(Y1(I,NJ+1)-Y1(I,NJ)+
     &		  Y1(I-1,NJ+1)-Y1(I-1,NJ))))

		  ELSE

C		  SOURCE TERM OF PRESSURE FOR Y-MOMENTUM - INSIDE

		  PV(I,J)=((0.5*(X1(I,J)-X1(I-1,J)+
     &		  X1(I,J-1)-X1(I-1,J-1)))*(0.5*(P(I,J-1)-
     &		  P(I,J+1))))

		  PU(I,J)=PU(I,J)+((0.5*(Y1(I,J)-Y1(I-1,J)+
     &		  Y1(I,J-1)-Y1(I-1,J-1)))*(0.5*(P(I,J-1)-
     &		  P(I,J+1))))

		  END IF

		END IF


		END DO
	END DO


c	PRINT*,((PV(I,J),I=2,NI+1),J=2,NJ+1)
c	STOP


c	STOP


C********************************************************************
C********************************************************************

C	Implementing the SIMPLE	algorithm

C	u-velocity

C	Applying the TDMA method to solve the discretized momentum equations

	MODE=1     !Solving the discretized x-momentum equation

	NeqV=(NI+2)*(NJ+2)	!Number of nodal points - includes known 
			        !and unknown values

	g=NeqV
	
	DO J=1,NJ+2 
    
        g=g-(NI+2)
        m=0

        	DO I=1,NI+2

                m=m+1 

		VXrec(g+m)=VX(I,J)

		END DO
	
	END DO

	Neq=(NI)*(NJ)


	CALL COEFF_TDMA(Neq,NeqV,vxrec,AP,AE,AW,AN,AS,
     &  PU,NI,NJ,MODE)

	g=NeqV
	
	DO J=1,NJ+2  
    
        g=g-(NI+2)

        m=0

        	DO I=1,NI+2

                m=m+1 

		Vx(I,J)=Vxrec(g+m)

		END DO
	
	END DO
c	print*,(vx(I,2),I=1,nI+2)
c	STOP
c	v-velocity

	MODE=2     !Solving the discretized y-momentum equation

	g=NeqV
	
	DO I=1,NI+2 
    
        g=g-(NJ+2)
        m=0

        	DO J=1,NJ+2

                m=m+1 

		Vyrec(g+m)=Vy(I,J)

		END DO
	
	END DO



C	WRITE(*,*)
C	PRINT*,'VELOCIDADE-y:'
C	WRITE(*,*)

	CALL COEFF_TDMA(Neq,NeqV,vyrec,AP,AE,AW,AN,AS,
     &  PV,NI,NJ,MODE)
	
	g=NeqV

	DO I=1,NI+2
    
        g=g-(NJ+2)
        m=0

        	DO J=1,NJ+2  

                m=m+1 

		Vy(I,J)=Vyrec(g+m)

c		IF (it==2) PRINT*,VY(I,J),I,J

		END DO
	
	END DO

c	PRINT*,(VY(I,1),I=2,NI+1)
c	STOP

C	Calculating the face velocity

	DO J=2,NJ+1

		DO I=2,NI

		IF (I==2) THEN

		FACE_U(2,J)=((1-ALFU)*FACE_U(2,J))-(0.5*(1-ALFU)
     &		*(VX_OLD(2,J)+VX_OLD(3,J)))+(0.5*(VX(2,J)+VX(3,J))
     &		)+(0.5*ALFU*(0.5*(Y1(2,J)-Y1(2,J-1)+Y1(1,J)-
     &		Y1(1,J-1)))*(P(3,J)-P(2,J))/AP(2,J))
     &		+(0.25*ALFU*(P(4,J)-P(2,J))*(0.5*(Y1(3,J)-Y1(3,J-1)
     &		+Y1(2,J)-Y1(2,J-1)))/AP(3,J))-(0.5*ALFU*(Y1(2,J)-
     &		Y1(2,J-1))*(P(3,J)-P(2,J))*((1./AP(3,J))+
     &		(1./AP(2,J))))

		ELSE

		  IF (I==NI) THEN

		  FACE_U(NI,J)=((1-ALFU)*FACE_U(NI,J))-(0.5*(1-ALFU)
     &		  *(VX_OLD(NI,J)+VX_OLD(NI+1,J)))+(0.5*(VX(NI,J)+
     &		  VX(NI+1,J)))+(0.25*ALFU*(P(NI+1,J)-P(NI-1,J))*(0.5
     &		  *(Y1(NI,J)-Y1(NI,J-1)+Y1(NI-1,J)-Y1(NI-1,J-1)))/
     &		  AP(NI,J))+(0.5*ALFU*(0.5*(Y1(NI+1,J)-Y1(NI+1,J-1)+
     &		  Y1(NI,J)-Y1(NI,J-1)))*(P(NI+2,J)-(0.5*(P(NI,J)+
     &		  P(NI+1,J))))/AP(NI+1,J))-(0.5*ALFU*(Y1(NI,J)-
     &		  Y1(NI,J-1))*(P(NI+1,J)-P(NI,J))*((1./AP(NI,J))+
     &		  (1./AP(NI+1,J))))

		  ELSE

		  FACE_U(I,J)=((1-ALFU)*FACE_U(I,J))-(0.5*(1-ALFU)
     &		  *(VX_OLD(I,J)+VX_OLD(I+1,J)))+(0.5*(VX(I,J)+
     &		  VX(I+1,J)))+(0.25*ALFU*(P(I+1,J)-P(I-1,J))*(0.5*(
     &		  Y1(I,J)-Y1(I,J-1)+Y1(I-1,J)-Y1(I-1,J-1)))/AP(I,J)
     &		  )+(0.25*ALFU*(0.5*(Y1(I+1,J)-Y1(I+1,J-1)+Y1(I,J)-
     &		  Y1(I,J-1)))*(P(I+2,J)-P(I,J))/AP(I+1,J))-(0.5*
     &		  ALFU*(Y1(I,J)-Y1(I,J-1))*(P(I+1,J)-P(I,J))*((1./
     &		  AP(I,J))+(1./AP(I+1,J))))

		  END IF

		END IF


		END DO

	VX(NI+2,J)=(1.5*VX(NI+1,J))-(0.5*VX(NI,J))

	VY(NI+2,J)=VY(NI+1,J)

	FACE_U(NI+1,J)=VX(NI+2,J)

	END DO


	DO I=2,NI+1

		DO J=2,NJ

		FACEJ_U(I,J)=(0.5*(VX(I,J)+VX(I,J+1)))

		IF (J==2) THEN

		FACE_V(I,2)=((1-ALFV)*FACE_V(I,2))-(0.5*(1-ALFV)
     &		*(VY_OLD(I,2)+VY_OLD(I,3)))+(0.5*(VY(I,2)+VY(I,3)))
     &		+(0.5*ALFV*(0.5*(X1(I,2)-X1(I-1,2)+X1(I,1)-X1(I-1,1))
     &		)*((0.5*(P(I,3)+P(I,2)))-P(I,2))/AP(I,2))+(0.25*ALFV
     &		*(P(I,4)-P(I,2))*(0.5*(X1(I,3)-X1(I-1,3)+X1(I,2)-
     &		X1(I-1,2)))/AP(I,3))-(0.5*ALFV*(X1(I,2)-X1(I-1,2))*
     &		(P(I,3)-P(I,2))*((1./AP(I,2))+(1./AP(I,3))))

		ELSE

		  IF (J==NJ) THEN

		  FACE_V(I,NJ)=((1-ALFV)*FACE_V(I,NJ))-(0.5*(1-ALFV)
     &		  *(VY_OLD(I,NJ)+VY_OLD(I,NJ+1)))+(0.5*(VY(I,NJ)+
     &		  VY(I,NJ+1)))+(0.25*ALFV*(0.5*(X1(I,NJ)-X1(I-1,NJ)
     &		  +X1(I,NJ-1)-X1(I-1,NJ-1)))*(P(I,NJ+1)-P(I,NJ-1))
     &		  /AP(I,NJ))+(0.5*ALFV*(0.5*(X1(I,NJ+1)-X1(I-1,NJ+1)
     &		  +X1(I,NJ)-X1(I-1,NJ)))*(P(I,NJ+1)-(0.5*(P(I,NJ)+
     &		  P(I,NJ+1))))/AP(I,NJ+1))-(0.5*ALFV*(X1(I,NJ)-
     &		  X1(I-1,NJ))*(P(I,NJ+1)-P(I,NJ))*((1./AP(I,NJ))+
     &		  (1./AP(I,NJ+1))))

		  ELSE

		  FACE_V(I,J)=((1-ALFV)*FACE_V(I,J))-(0.5*(1-ALFV)
     &		  *(VY_OLD(I,J)+VY_OLD(I,J+1)))+(0.5*(VY(I,J)+
     &		  VY(I,J+1)))+(0.25*(0.5*(X1(I,J)-X1(I-1,J)+
     &		  X1(I,J-1)-X1(I-1,J-1)))*(P(I,J+1)-P(I,J-1))/
     &		  AP(I,J))+(0.25*(P(I,J+2)-P(I,J))*(0.5*(X1(I,J+1)
     &		  -X1(I-1,J+1)+X1(I,J)-X1(I-1,J)))/AP(I,J+1))-(0.5
     &		  *(X1(I,J)-X1(I-1,J))*(P(I,J+1)-P(I,J))*((1./
     &		  AP(I,J+1))+(1./AP(I,J))))

		  END IF

		END IF

		END DO

	END DO

c	Pressure correction

c	Mass imbalance

C	WRITE(*,*)
C	PRINT*,'Termo b'

	SSUM=0.

C	PRINT*,(FACEJ_U(I,2),I=2,NI+1)

	DO J=2,NJ+1
		DO I=2,NI+1

		w=0.
		e=0.
		s=0.
		n=0.

		n=(ROO(I,J)*FACE_V(I,J)*(X1(I,J)-X1(I-1,J)))+
     & 		(ROO(I,J)*FACEJ_U(I,J)*(Y1(I,J)-Y1(I-1,J)))

		s=(ROO(I,J)*FACE_V(I,J-1)*(X1(I,J-1)-X1(I-1,J-1)))+
     &		(ROO(I,J)*FACEJ_U(I,J-1)*(Y1(I,J-1)-Y1(I-1,J-1)))

		IF (I==NI+1) THEN

		e=(1.5*ROO(I,J)*VX(NI+1,J)*(Y1(NI+1,J)-Y1(NI+1,J-1)))
	 
		w=(1.5*ROO(I,J)*FACE_U(NI,J)*(Y1(NI,J)-Y1(NI,J-1)))

		ELSE

		e=(ROO(I,J)*FACE_U(I,J)*(Y1(I,J)-Y1(I,J-1)))

		w=(ROO(I,J)*FACE_U(I-1,J)*(Y1(I-1,J)-Y1(I-1,J-1)))
	
		END IF

		ERROR(I,J)=w-e+s-n

		SSUM=SSUM+ABS(ERROR(I,J))

		END DO
	END DO

	limite=1.1E-24

	DO J=2,NJ+1
		DO I=2,NI+1

		limite=amax1(limite,ABS(ERROR(I,J)))


		END DO
	END DO

c	Cálculo dos coeficientes da eq discretizada da
c	correção de pressão

	DO J=2,NJ+1
		DO I=2,NI
c
c		SOUTH WALL
C
		IF (J==2) THEN

		ASP(I,J)=0.

		ELSE

		ASP(I,J)=(0.5*ROO(I,J)*((X1(I,J-1)-X1(I-1,J-1))
     &		**(2))*(ALFV/(1.-ALFV))*((1./AP(I,J))+(1./AP(I,J-1))))

		END IF

C		NORTH WALL

		IF (J==NJ+1) THEN

		ANP(I,J)=0.

		ELSE

		ANP(I,J)=(0.5*ROO(I,J)*((X1(I,J)-X1(I-1,J))
     &		**(2))*(ALFV/(1.-ALFV))*((1./AP(I,J+1))+(1./AP(I,J))))

		END IF

C		INLET - WEST BOUNDARY

		IF (I==2) THEN

		AWP(I,J)=0.

		ELSE

		AWP(I,J)=(0.5*ROO(I,J)*((Y1(I-1,J)-Y1(I-1,J-1))
     &		**(2))*(ALFU/(1.-ALFU))*((1./AP(I,J))+(1./AP(I-1,J))))		 

		END IF

		AEP(I,J)=(0.5*ROO(I,J)*((Y1(I,J)-Y1(I,J-1))
     &		**(2))*(ALFU/(1.-ALFU))*((1./AP(I,J))+(1./AP(I+1,J))))
	

		APP(I,J)=(AEP(I,J)+AWP(I,J)+ASP(I,J)+ANP(I,J))	

		END DO
	END DO


C	OUTLET - EAST BOUNDARY

	DO J=2,NJ+1

	  IF (J==2) THEN

	  ASP(NI+1,J)=0.

	  ELSE

	  ASP(NI+1,J)=(0.5*ROO(I,J)*((X1(I,J-1)-X1(I-1,J-1))
     &	  **(2))*(ALFV/(1.-ALFV))*((1./AP(I,J))+(1./AP(I,J-1))))

	  END IF

	  IF (J==NJ+1) THEN

	  ANP(NI+1,J)=0.

	  ELSE

	  ANP(NI+1,J)=(0.5*ROO(I,J)*((X1(I,J)-X1(I-1,J))
     &	  **(2))*(ALFV/(1.-ALFV))*((1./AP(I,J+1))+(1./AP(I,J))))

 	  END IF

	AEP(NI+1,J)=0.

	AWP(NI+1,J)=(1.5*ROO(I,J)*((Y1(I-1,J)-Y1(I-1,J-1))
     &	**(2))*(ALFU/(1.-ALFU))*((1./AP(I,J))+(1./AP(I-1,J))))
     &  -(1.5*ROO(I,J)*((0.5*(Y1(I-1,J)-Y1(I-1,J-1)+Y1(I,J)-
     &	Y1(I,J-1)))**(2))*(ALFU/(1.-ALFU))/AP(I,J))

	APP(NI+1,J)=(1.5*ROO(I,J)*((Y1(I-1,J)-Y1(I-1,J-1))
     &	**(2))*(ALFU/(1.-ALFU))*((1./AP(I,J))+(1./AP(I-1,J))))
     &  +(1.5*ROO(I,J)*((0.5*(Y1(I-1,J)-Y1(I-1,J-1)+Y1(I,J)-
     &	Y1(I,J-1)))**(2))*(ALFU/(1.-ALFU))/AP(I,J))+ANP(NI+1,J)+
     &	ASP(NI+1,J)	

	END DO




C	PRESSURE CORRECTION FIELD

	MODE=3

C	Preenchendo a matriz dos coeficientes

	NeqP=NeqV

	g=NeqP

	DO J=1,NJ+2 
  
        g=g-(NI+2)
        m=0

        	DO I=1,NI+2

                m=m+1 

		Prec(g+m)=Pcor(I,J)

		END DO
	
	END DO


	CALL COEFF_TDMA(NeqP,NeqP,Prec,APP,AEP,AWP,ANP,ASP,
     &  ERROR,NI,NJ,MODE)
	

c	Resolução da equação discretizada de correção de pressão


C	PRINT*,'PRESSURE CORRECTION'
C	WRITE(*,*)


	DO J=1,NJ+2

	g=NeqP-(J*(NI+2))

	m=0

		DO I=1,NI+2
		
		m=m+1

		Pcor(I,J)=Prec(g+m)

		END DO

	END DO

	DO I=2,NI+1
		DO J=2,NJ+1

		P(I,J)=P(I,J)+(ALFP*(Pcor(I,J)))

		END DO
	END DO


c	VELOCITY CORRECTIONS

C	CELL CENTER VELOCITY

	DO J=2,NJ+1
		DO I=2,NI+1
		
		VX_OLD(I,J)=VX(I,J)

		VX(I,J)=(ALFU*(0.5*(Y1(I,J)-Y1(I,J-1)
     &		+Y1(I-1,J)-Y1(I-1,J-1)))*(0.5*(Pcor(I-1,J)-
     &		Pcor(I+1,J)))/AP(I,J))+VX(I,J)

		VY_OLD(I,J)=VY(I,J)	

		VY(I,J)=(ALFV*(0.5*(X1(I,J)-X1(I-1,J)
     &		+X1(I,J-1)-X1(I-1,J-1)))*(0.5*(Pcor(I,J-1)-
     &		Pcor(I,J+1)))/AP(I,J))+VY(I,J)	

		END DO
	END DO

c	print*,'vy1',vy(25,8)

C	FACE VELOCITY

	DO J=2,NJ+1
		DO I=2,NI

		FACE_U(I,J)=(ALFU*(Y1(I,J)-Y1(I,J-1))
     &		*(0.5*(Pcor(I,J)-Pcor(I+1,J)))*((1./AP(I+1,J))+
     &		(1./AP(I,J))))+FACE_U(I,J)

		END DO
	END DO

	DO I=2,NI+1
		DO J=2,NJ

		FACE_V(I,J)=(ALFV*(X1(I,J)-X1(I-1,J))
     &		*(0.5*(Pcor(I,J)-Pcor(I,J+1)))*((1./AP(I,J+1))+
     &		(1./AP(I,J))))+FACE_V(I,J)

		FACEJ_U(I,J)=(0.5*(VX(I,J)+VX(I,J+1)))

		END DO
	END DO

	VY(12,2)=VY(12,1)

C 	Saving the result of the previous iteration

	DO I=2,NI+1
		DO J=2,NJ+1

		MASS_it(I,J)=MASS(I,J)

		END DO
	END DO

C 	Calculating the mass fraction

	CALL MASS_FRACTION(NI,NJ,X1,Y1,X,Y,VX,VY,ROO,VISC,FACE_U,
     &  FACE_V,FACEJ_U,MASS,DAB)


C	Adicionando coeficiente de relaxação

	DO I=2,NI+1
		DO J=2,NJ+1

		MASS(I,J)=MASS(I,J)*0.1+MASS_it(I,J)*(1-0.1)

		END DO
	END DO	


C	Calculating the density and viscosity of the mixture


	if (it>60000) then
	

	 DO I=2,NI+1
		DO J=2,NJ+1

		ROO(I,J)=(1./((MASS(I,J)/ROA)+((1.-MASS(I,J))/ROB)))

		VISC(I,J)=(1./((MASS(I,J)/VISA)+((1.-MASS(I,J))/VISB)))
c
                END DO

        END DO

	end if

	if (it<=60000 .and. mod(it,3000)==0) then
	

	 DO I=2,NI+1
		DO J=2,NJ+1

		ROO(I,J)=(1./((MASS(I,J)/ROA)+((1.-MASS(I,J))/ROB)))

		VISC(I,J)=(1./((MASS(I,J)/VISA)+((1.-MASS(I,J))/VISB)))
c
               END DO

        END DO

	end if


C	Calculating the error

	Error_mass=0.

	DO J=2,NJ+1
		DO I=2,NI+1

c		
		difference(I,J)=MASS(I,J)-MASS_it(I,J)

		Error_mass=Error_mass+ABS(difference(I,J))

		END DO
	END DO
cc
	print*,'ERR_simple=',SSUM,'  ERR_Ya=',
     &		Error_mass,'ITER=',it        

c	print*,'ERR_simple=',SSUM,'ITER=',it     

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC


		if (limite>1.e+23 .or. Error_mass>1.e+23) then
	
		PRINT*,'DIVERGIU'
		EXIT

		end if

	IF ((it/vtk)==1) then

	CALL PLOTTING(NI,NJ,X1,Y1,ROO,VISC,VX,VY,P,X,Y,MASS,STRF,VORT)

	vtk=vtk+vtk1

	end if


	IF(it==1) THEN
	WRITE(13,100)
100 	FORMAT(' ITER',10X,'Simple-error',10X,'Ya-error')
c100 	FORMAT(' ITER',10X,'Simple-error')
	END IF

C	IF (it>9999) then

110 	WRITE(13,120)IT,SSUM,Error_mass
c110 	WRITE(13,120)IT,SSUM
120 	FORMAT(I6,4E20.8)

C	END IF



	P(1,1)=P(1,2)
	P(1,NJ+2)=P(1,NJ+1)
	P(NI+2,1)=P(NI+2,2)
	P(NI+2,NJ+2)=P(NI+2,NJ+1)

	END DO


	CLOSE(UNIT=13)


	RETURN
	END


ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


      SUBROUTINE TDMA(N,A,X,NI,NJ,c,E)

C****************************************************************
c This program solves a linear system using the TDMA algorithm.
c The input file tdma.dat, consists of the augmented matrix
c of the coeficients and the independent vector.
c The used must specify through the terminal the number of equations
c of the system.

C*****************************************************************
        Parameter (i_max=8000,j_max=1000)
        Real A(i_max,j_max),X(i_max),X1(i_max)
        Real L(i_max),U(i_max),D(i_max),B(i_max),D_prime(i_max)
        Real B_prime(i_max),U_prime(i_max)
        Integer I,J,idx,N,g,E,F,resultado,m,summ,c
        Real ABSPIV,TEMP,MULT


C	X-MOMENTUM AND PRESSURE CORRECTION

	IF (c==1 .or. c==3) THEN

	g=N
	
	g=g-((E-1)*NI)
	m=0

	  DO F=2,NI+1
	  m=m+1
	    
c**************************************************
c Identifying elements of the diagonal

		   d(m)=a(g+m,g+m)
		   
c**************************************************
c Identifying elements upper of the diagonal
 		if (g+m/=N) then

                   u(m)=a(g+m,g+m+1)

		ELSE

		   u(m)=0.

		end if

c**************************************************
c Identifying elements lower of the diagonal

		IF ((g+m-1)<1) THEN

		   l(m)=0.

		ELSE

		   l(m)=a(g+m,g+m-1)

		END IF


c**************************************************
c Identifying the independent elements
 

                   b(m)=a(g+m,n+1)

c		if (c==3) print*,b(m),m

	  END DO

c*************************************************
c Upper elements updated 

             u_prime(1)=u(1)/d(1)
             DO I=2,Ni
                  u_prime(i)=u(i)/(d(i)-(u_prime(i-1)*l(i)))
             END DO  

C**************************************************
c Independent elements updated 

             b_prime(1)=b(1)/d(1)
             DO I=2,Ni
                  b_prime(i)=(b(i)+b_prime(i-1)*l(i))
     & /(d(i)-u_prime(i-1)*l(i))
             END DO  

c**************************************************
c Solution vector using backward substitution

c               X1(m)=b_prime(m)
		X1(M+1)=0.
	
             DO I=m,1,-1
                
                X1(I)=b_prime(i)+u_prime(i)*X1(i+1)

             END DO
		
		  m=0
	
	  	  DO F=2,NI+1
	  	    m=m+1
	    	  
		    X(g+m)=X1(m)
		
		  end do

	ELSE

C	Y-MOMENTUM

	g=N
	
	g=g-((E-1)*NJ)
	m=0

	  DO F=2,NJ+1
	  m=m+1
	    
c**************************************************
c Identifying elements of the diagonal

		   d(m)=a(g+m,g+m)
		   
c**************************************************
c Identifying elements upper of the diagonal
 		if (g+m/=N) then

                   u(m)=a(g+m,g+m+1)

		ELSE

		   u(m)=0.

		end if

c**************************************************
c Identifying elements lower of the diagonal

		IF ((g+m-1)<1) THEN

		   l(m)=0.


		ELSE

		   l(m)=a(g+m,g+m-1)

		END IF


c**************************************************
c Identifying the independent elements
 

                   b(m)=a(g+m,n+1)

c		if (c==3) print*,b(m),m

	  END DO

c*************************************************
c Upper elements updated 

             u_prime(1)=u(1)/d(1)
             DO J=2,NJ
                  u_prime(J)=u(J)/(d(J)-(u_prime(J-1)*l(J)))
             END DO  

C**************************************************
c Independent elements updated 

             b_prime(1)=b(1)/d(1)
             DO J=2,NJ
                  b_prime(J)=(b(J)+b_prime(J-1)*l(J))
     & /(d(J)-u_prime(J-1)*l(J))
             END DO  

c**************************************************
c Solution vector using backward substitution

c               X1(m)=b_prime(m)
		X1(M+1)=0.
	
             DO J=m,1,-1
                
                X1(J)=b_prime(J)+u_prime(J)*X1(J+1)

             END DO
		
		  m=0
	
	  	  DO F=2,NJ+1
	  	    m=m+1
	    	  
		    X(g+m)=X1(m)
		
		  end do


	END IF

	END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

    	SUBROUTINE COEFF_TDMA(n,NV,x1,AP,AE,AW,AN,AS,S,NI,NJ,
     &  mode)

        Parameter (i_max=8000,j_max=1000)
	REAL,DIMENSION(i_max,j_max) :: S,A,AP,AE,AW,AN,AS
	REAL,DIMENSION(i_max) :: X1,X2
	INTEGER g,g1,m,NI,NJ,mode

	g=n

	DO J=1,n+1
		DO I=1,n

		A(I,J)=0.

		END DO
	END DO


	DO I=1,n

		X2(I)=0.

	END DO


C*****************************************************
C*****************************************************

C	X-MOMENTUM (x=1) PRESSURE CORRECTION (x=3) EQUATIONS

	IF (mode==1 .or. mode==3) THEN

	DO J=2,NJ+1  
    
        g=g-NI
	g1=NV-(J*(NI+2))
        m=0

        	DO I=2,NI+1
		
                m=m+1 	
		
		A(g+m,g+m)=AP(I,J)
			
		IF (J==2) THEN

			IF (I==2) THEN

			A(g+m,n+1)=S(I,J)+
     &			(AN(I,J)*X1(g1+I-(NI+2)))

			A(g+m,g+m+1)=AE(I,J)

			ELSE

			  IF (I==NI+1) THEN
			
			  A(g+m,g+m-1)=AW(I,J)

			  A(g+m,n+1)=S(I,J)+
     &			  (AN(I,J)*(X1(g1+I-(NI+2))))
c     &			  +(AE(I,J)*(X1(g1+I+1)))

			  ELSE

			  A(g+m,g+m-1)=AW(I,J)

			  A(g+m,g+m+1)=AE(I,J)

			  A(g+m,n+1)=S(I,J)+
     &			  (AN(I,J)*(X1(g1+I-(NI+2))))

			  END IF

			END IF

		ELSE
		
		IF (J==NJ+1) THEN

			IF (I==2) THEN

			A(g+m,g+m+1)=AE(I,J)

			A(g+m,n+1)=S(I,J)+
     &			(AS(I,J)*X1(g1+I+(NI+2)))
			
			ELSE

			 IF (I==NI+1) THEN

			 A(g+m,g+m-1)=AW(I,J)

			 A(g+m,n+1)=S(I,J)+
     &			 (AS(I,J)*(X1(g1+I+(NI+2))))
c     &			 +(AE(I,J)*(X1(g1+I+1)))


			 ELSE

			 A(g+m,g+m-1)=AW(I,J)

			 A(g+m,g+m+1)=AE(I,J)
			
			 A(g+m,n+1)=S(I,J)+
     &			 (AS(I,J)*(X1(g1+I+(NI+2))))

			 END IF

			END IF

		ELSE
		
		IF (J/=2 .AND. J/=NJ+1) THEN

		  IF (I==2) THEN

			A(g+m,n+1)=S(I,J)+
     &			(X1(g1+I+(NI+2))*AS(I,J))+(AN(I,J)*
     &			X1(g1+I-(NI+2)))

			A(g+m,g+m+1)=AE(I,J)
			
			ELSE

			 IF (I==NI+1) THEN

			 A(g+m,n+1)=S(I,J)+
     &			 (X1(g1+I+(NI+2))*AS(I,J))+(AN(I,J)
     &			 *X1(g1+I-(NI+2)))
c     &			 +(AE(I,J)*X1(g1+I+1))

			 A(g+m,g+m-1)=AW(I,J)

			 ELSE
			
			 A(g+m,n+1)=S(I,J)+
     &			 (X1(g1+I+(NI+2))*AS(I,J))+(AN(I,J)
     &			 *X1(g1+I-(NI+2)))

			 A(g+m,g+m-1)=AW(I,J)

			 A(g+m,g+m+1)=AE(I,J)

			 END IF
		  END IF

		END IF

		END IF

		END IF

		X2(g+m)=x1(g1+I)

                END DO

        CALL TDMA(N,A,X2,ni,nj,mode,J)

             DO I=m,1,-1

		x1(g1+I+1)=X2(g+I)
c		print*,x1(g1+i+1),i,j
		
	     END DO

      END DO


C	Y-MOMENTUM

	ELSE

	DO I=2,NI+1  
    
        g=g-NJ
	g1=NV-(I*(NJ+2))
        m=0

        	DO J=2,NJ+1
		
                m=m+1 	
		
		A(g+m,g+m)=AP(I,J)
			
		IF (J==2) THEN

			IF (I==2) THEN

			A(g+m,n+1)=S(I,J)+
     &			(AE(I,J)*X1(g1+J-(NJ+2)))

			A(g+m,g+m+1)=AN(I,J)

			ELSE

			  IF (I==NI+1) THEN
			
			  A(g+m,g+m+1)=AN(I,J)

			  A(g+m,n+1)=S(I,J)+
     &			  (AW(I,J)*(X1(g1+J+(NJ+2))))

			  ELSE

			  A(g+m,g+m+1)=AN(I,J)

			  A(g+m,n+1)=S(I,J)+
     &			  (AW(I,J)*(X1(g1+J+(NJ+2))))+
     &			  (AE(I,J)*(X1(g1+J-(NJ+2))))

			  END IF

			END IF

		ELSE
		
		IF (J==NJ+1) THEN

			IF (I==2) THEN

			A(g+m,g+m-1)=AS(I,J)

			A(g+m,n+1)=S(I,J)+
     &			(AE(I,J)*X1(g1+J-(NJ+2)))
			
			ELSE

			 IF (I==NI+1) THEN

			 A(g+m,g+m-1)=AS(I,J)

			 A(g+m,n+1)=S(I,J)+
     &			 (AW(I,J)*(X1(g1+J+(NJ+2))))


			 ELSE

			 A(g+m,g+m-1)=AS(I,J)
			
			 A(g+m,n+1)=S(I,J)+
     &			 (AW(I,J)*(X1(g1+J+(NJ+2))))+
     &			 (AE(I,J)*(X1(g1+J-(NJ+2))))

			 END IF

			END IF

		ELSE
		
		IF (J/=2 .AND. J/=NJ+1) THEN

		  IF (I==2) THEN

			A(g+m,n+1)=S(I,J)+
     &			(X1(g1+J-(NJ+2))*AE(I,J))

			A(g+m,g+m+1)=AN(I,J)

			A(g+m,g+m-1)=AS(I,J)
			
			ELSE

			 IF (I==NI+1) THEN

			 A(g+m,n+1)=S(I,J)+
     &			 (AW(I,J)*X1(g1+J+(NJ+2)))	 

			 A(g+m,g+m-1)=AS(I,J)

			 A(g+m,g+m+1)=AN(I,J)

			 ELSE
			
			 A(g+m,n+1)=S(I,J)+
     &			 (X1(g1+J-(NJ+2))*AE(I,J))+(AW(I,J)
     &			 *X1(g1+J+(NJ+2)))

			 A(g+m,g+m-1)=AS(I,J)

			 A(g+m,g+m+1)=AN(I,J)

			 END IF
		  END IF

		END IF

		END IF

		END IF

		X2(g+m)=x1(g1+J)

                END DO

        CALL TDMA(N,A,X2,ni,nj,mode,I)

             DO J=m,1,-1

		x1(g1+J+1)=X2(g+J)
c		PRINT*,X1(G1+J+1),I,J
		
	     END DO

      END DO




	END IF


	RETURN
	END


CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

	SUBROUTINE VORTICITY(NI,NJ,VX,VY,X,Y,X1,Y1,VORT,itt)

        Parameter (i_max=8000,j_max=1000)

	INTEGER :: I,J,NI,NJ,g,m,it
	REAL,DIMENSION(i_max,j_max) :: VX,VY,X,Y,X1,Y1,
     &  VORT
	REAL :: w,e,s,n


C	Components for the Vorticity Equation

	DO J=2,NJ+1
		DO I=2,NI+1

		VORT(I,J)=0.

		IF (I==2) THEN

		w=-(VY(1,J)/(0.5*(X1(I,J)-X1(I-1,J)+X1(I,J-1)
     &		   -X1(I-1,J-1))))

		ELSE

		w=-(0.5*(VY(I,J)+VY(I-1,J)))/(0.5*(X1(I,J)-X1(I-1,J)+
     &		   X1(I,J-1)-X1(I-1,J-1)))

		END IF	

		IF (I==NI+1) THEN	

		e=+(VY(NI+2,J)/(0.5*(X1(I,J)-X1(I-1,J)+X1(I,J-1)
     &		   -X1(I-1,J-1))))

		ELSE

		e=+(0.5*(VY(I+1,J)+VY(I,J)))/(0.5*(X1(I,J)-X1(I-1,J)
     &		   +X1(I,J-1)-X1(I-1,J-1)))

		END IF

		IF (J==NJ+1) THEN

		n=-(VX(I,NJ+2))/(0.5*(Y1(I,J)-Y1(I,J-1)+Y1(I-1,J)
     &		   -Y1(I-1,J-1)))

		ELSE

		n=-(0.5*(VX(I,J+1)+VX(I,J)))/(0.5*(Y1(I,J)-Y1(I,J-1)
     &		   +Y1(I-1,J)-Y1(I-1,J-1)))
		
		END IF

		IF (J==2) THEN

		s=+(VX(I,1)/(0.5*(Y1(I,J)-Y1(I,J-1)+Y1(I-1,J)
     &		   -Y1(I-1,J-1))))

		ELSE

		s=+(0.5*(VX(I,J-1)+VX(I,J)))/(0.5*(Y1(I,J)-Y1(I,J-1)
     &		   +Y1(I-1,J)-Y1(I-1,J-1)))

		END IF

		VORT(I,J)=w+e+n+s

		END DO
	END DO

	RETURN
	END


CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
	SUBROUTINE STREAMFUNCTION(NI,NJ,VX,VY,X,Y,X1,Y1,STRF,
     &  VORT)

        Parameter (i_max=8000,j_max=1000)

	INTEGER :: I,J,NI,NJ,g,m,it,MODE,NeqVORT
	REAL,DIMENSION(i_max,j_max) :: VX,VY,X,Y,X1,Y1,STRF,AE,AW,
     &  AN,AS,SS,AP,A_TDMA,VORT,old_strf,RES_STRF
	REAL :: STRFrec(i_max),minimum,UIN,w,e,s,n,RES_SUMM



	DO J=2,NJ+1
		DO I=2,NI+1

		AE(I,J)=(Y1(I,J)-Y1(I,J-1))/(0.5*(X(I+1,J)-
     &		X(I,J)+X(I+1,J-1)-X(I,J-1)))

		AW(I,J)=(Y1(I,J)-Y1(I,J-1))/(0.5*(X(I,J)-
     &		X(I-1,J)+X(I,J-1)-X(I-1,J-1)))

		AN(I,J)=(X1(I,J)-X1(I-1,J))/(0.5*(Y(I,J+1)-
     &		Y(I,J)+Y(I-1,J+1)-Y(I-1,J)))

		AS(I,J)=(X1(I,J)-X1(I-1,J))/(0.5*(Y(I,J)-
     &		Y(I,J-1)+Y(I-1,J)-Y(I-1,J-1)))

		AP(I,J)=AE(I,J)+AW(I,J)+AN(I,J)+AS(I,J)

		SS(I,J)=VORT(I,J)*(0.5*(X1(I,J)-X1(I-1,J)+
     &		X1(I,J-1)-X1(I-1,J-1)))*(0.5*(Y1(I,J)-Y1(I,J-1)+
     &		Y1(I-1,J)-Y1(I-1,J-1)))

		END DO
	END DO

C	STREAM FUNCTION FIELD

	MODE=1

	NeqSTRF=(NI+2)*(NJ+2)

	DO it=1,4000
	
	RES_SUMM=0.
c
	g=NeqSTRF
c
	DO J=1,NJ+2
c
       g=g-(NI+2)
       m=0

        	DO I=1,NI+2
        	
        	IF(it==1) OLD_STRF(i,j)=0.

                m=m+1

		STRFrec(g+m)=STRF(I,J)

		END DO

	END DO
	
	Neq=(NI)*(NJ)

	CALL COEFF_TDMA(Neq,NeqSTRF,STRFrec,AP,
     &  AE,AW,AN,AS,SS,NI,NJ,MODE)


	  DO J=1,NJ+2

	  g=NeqSTRF-(J*(NI+2))

	  m=0

		DO I=1,NI+2

		m=m+1

		STRF(I,J)=STRFrec(g+m)

		END DO

	  END DO

           DO J=2,NJ+1
              DO I=2,NI+1
              
              RES_STRF(I,J)=STRF(I,J)-OLD_STRF(I,J)

              RES_SUMM=RES_SUMM+ABS(RES_STRF(I,J))

              OLD_STRF(I,J)=STRF(I,J)

              END DO
           END DO
           
        PRINT*,'Residual Stream-Function=',RES_SUMM,'     ITERATION=',IT
        
        IF (RES_SUMM==0.) EXIT
        
	END DO

C       ASSUMING THAT STREAM-FUNCTION VALUES AT THE CORNERS ARE EQUAL
C       TO THEIR NEIGHBORS

	STRF(1,1)=STRF(2,1)
	STRF(1,NJ+2)=STRF(2,NJ+2)
	STRF(NI+2,1)=STRF(NI+2,2)
	STRF(NI+2,NJ+2)=STRF(NI+2,NJ+1)


	RETURN

	END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

	SUBROUTINE CONVERTING(NI,NJ,P,X1,Y1,X,Y,ROO,VX,VY,STRF)

        Parameter (i_max=8000,j_max=1000)
	INTEGER :: I,J
	REAL,DIMENSION(i_max,j_max) :: P,X,Y,X1,Y1,ROO,VX,VY,STRF

	DO I=1,NI+2
		DO J=1,NJ+2

		P(I,J)=(P(I,J)/ROO(I,J))*1.E-6
		VX(I,J)=VX(I,J)*1.E-3
		VY(I,J)=VY(I,J)*1.E-3
		X(I,J)=10*X(I,J)
		Y(I,J)=10*Y(I,J)
		X1(I,J)=10*X1(I,J)
		Y1(I,J)=10*Y1(I,J)
		STRF(I,J)=STRF(I,J)*1.E-6		
		
		END DO
	END DO

	RETURN
	END



CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

	SUBROUTINE PLOTTING(NI,NJ,X1,Y1,ROO,VISC,VX,VY,P,X,Y,MASS,
     &  STRF,VORT)

        Parameter (i_max=8000,j_max=1000)
	INTEGER :: I,J,NPOINT,NCELL,A,B,C,D,e,f3
	REAL,DIMENSION(i_max,j_max) :: ROO,VX,VY,P,VISC,X1,Y1,
     &  X,Y,MASS,STRF,VORT

c	Pressure Field and other scalars

	NPOINT=(NI+2)*(NJ+2)
	NCELL=(NI+1)*(NJ+1)
	f3=NCELL*5	
	OPEN(UNIT=23,FILE='dispersion.vtk')

	WRITE(23,'(A,3I6)')'# vtk DataFile Version 5.4'
        WRITE(23,'(A41)') 'Uniform Nozzle Grid' 
        WRITE(23,'(A,3I6)') 'ASCII'
        WRITE(23,'(A,3I6)') 'DATASET UNSTRUCTURED_GRID'
	WRITE(23,'(A,1I6,A)') 'POINTS', NPOINT,' float'

        DO I=1,NI+2
             DO J=1,NJ+2
                WRITE(23,'(2F7.2,I2)') X(I,J), Y(I,J),0
             END DO
        END DO  
      
	WRITE(23,*)

        WRITE(23,'(A,2I10)')'CELLS',NCELL,f3

	A=0.
	B=A+1
	C=NJ+2
	D=C+1
	e=1
	DO I=1,(NI+1)*(NJ+1)
		IF (e==(NJ+1)) THEN
			WRITE(23,'(5I6)') 4,A,C,D,B
			e=1.
			A=A+2
			B=B+2
			C=C+2
			D=D+2

		ELSE
			WRITE(23,'(5I6)') 4,A,C,D,B
			A=A+1
			B=B+1
			C=C+1
			D=D+1
			e=e+1

		END IF
	END DO

        WRITE(23, *) 
        WRITE(23,'(A,1I5)')'CELL_TYPES',NCELL
        WRITE(23,'(4000I2)')(9,I=1,NCELL)
        WRITE(23, *) 
	WRITE(23,'(A,1I5)') 'POINT_DATA',NPOINT

      write (23,'(A,1I2)') 'SCALARS SPECIE_A float', 1
      write (23,'(A)') 'LOOKUP_TABLE default'
         DO I=1,NI+2
             DO J=1,NJ+2
                write(23,'(100E28.16)') MASS(I,J)
             END DO
         END DO

         
      write (23,'(A,1I2)') 'SCALARS PRESSURE float', 1
      write (23,'(A)') 'LOOKUP_TABLE default'
         DO I=1,NI+2
             DO J=1,NJ+2
                write(23,'(100E28.16)') P(I,J)
             END DO
         END DO

      write (23,'(A,1I2)') 'SCALARS DENSITY float', 1
      write (23,'(A)') 'LOOKUP_TABLE default'
         DO I=1,NI+2
             DO J=1,NJ+2
                write(23,'(100E28.16)') ROO(I,J) 
             END DO
         END DO

      write (23,'(A,1I2)') 'SCALARS VISCOSITY float', 1
      write (23,'(A)') 'LOOKUP_TABLE default'
         DO I=1,NI+2
             DO J=1,NJ+2
                write(23,'(100E28.16)') VISC(I,J) 
             END DO
         END DO

	WRITE(23,'(A)') 'VECTORS VELOCITY float'
         DO I=1,NI+2
            DO J=1,NJ+2
               WRITE (23,'(100E28.16)') VX(I,J), VY(I,J), 0.0
            END DO
         END DO
         
        WRITE(23,'(A)') 'VECTORS VORTICITY float'
         DO I=1,NI+2
            DO J=1,NJ+2
               WRITE(23,'(100E28.16)') 0.0, 0.0, VORT(I,J)
            END DO
         END DO

      WRITE (23,'(A,1I2)') 'SCALARS STREAM_FUNCTION float', 1
      WRITE (23,'(A)') 'LOOKUP_TABLE default'
         DO I=1,NI+2
             DO J=1,NJ+2
                write(23,'(100E28.16)') STRF(I,J) 
             END DO
         END DO

  
        close(23)

        RETURN
        END

