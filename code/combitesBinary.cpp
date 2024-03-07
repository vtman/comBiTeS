#include <chrono>

#include <fstream>

#include <immintrin.h>
#include <smmintrin.h>
#include <emmintrin.h>
#include <string.h>

#include <omp.h>

//#define WIN32 1

#ifndef WIN32
#define _aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define _aligned_free(ptr) free(ptr)
#endif // !WIN32


const int MIN_START_LEVEL = 1;
const int SIZE_INFO = 4;
const int MAX_ROWS = 30;

class compSS {
public:
	compSS(int maxRows_in);
	~compSS();

	int printSIMDsolution(int iU);
	int printSolutionLetters(int iU);
	int openNewFile();
	int prepareFileMiddleName();
	int processLocalMatrix(int iU);
	int initiateLocalMatrix();
	int countBits(__m128i* vm);
	int printVector128(__m128i* vm);
	int findNonZeroElement(__m128i* vm);
	int findOneIn128(__m128i mA, int iStart);
	int processLevel(int nLevel_in);
	int formOriginalMatrix();
	int sortOriginalMatrix();
	int prepareSeed();
	int findSeedLen();
	int printSeed();
	int findVariables();
	int setM();
	int printMatrixOriginal();
	int printMatrixSorted();
	int printMatrixLocal(int iU);
	int printCandidateMatrix(int n);

	int freeMemory();

	long long iCount;

	int nMatrixRows, maxElementsInRow;
	int *matrixOriginal, *matrixSorted;
	__m128i* mMatrixOneOriginal, * mMatrixGapOriginal, * mMatrixOneSorted, * mMatrixGapSorted;
	__m128i* mRowTemp, * mRowFull, * mResultOne, * mResultGap;
	int* vecMatrixSortedInfo, *vecMatrixOriginalInfo;
	int * vIndActiveState;
	int iLevel;
	int* matColGap, * matColOne;
	int* vIndexRowGap, * vIndexRowOne;
	int* vCandidateRows;

	int* matrixLocal;
	__m128i* mMatrixLocalOne, *mMatrixLocalGap;
	__m128i* mPreviousOne, *mPreviousGap;
	int indSolution;

	bool isPrint;

	int* vPositionGap, * vPositionOne;
	int* vIndGapReverse, * vIndOneReverse;
	__m128i* mPosOne, * mPosGap;

	__m128i mShift32[32];

	FILE* fo;
	int *vIndexRows;
	char outputFileName[1000], outputFolderPrefix[1000], seedNameHex[1000];
	char seedLetterBefore[1000], seedLetterAfter[1000];
	int seedLen, seedWeight, *vSeedBefore, *vSeedLetter, nVar, nVar128;
	
	int nRowGap, nRowOne, n32, nRows, maxRows;

	long long elapsedTime;

	__m128i mOne[128], mZero;

	int currentLevel, indProblem, seedLenOrig;
	int *vSeedOrig, *vSeed;
};


class globalCompass {
public:
	globalCompass();
	~globalCompass();
	int printUsage();
	int findProblems();

	int startProcessing(int nA, char **vA);

	int seedLen, seedWeight, * vSeed, *vSeedExtend, nProblems, nThreads;
	int *vCountOnes, *vIndNonZero;
	int* viProblemState;
	int nBest, startLevel;
	char outputFolderPrefix[1000];

	compSS** vecProblems;
};

globalCompass::globalCompass() {
	nProblems = 0;
	nThreads = 0;
	vSeed = nullptr;
	vSeedExtend = nullptr;
	vCountOnes = nullptr;
	vIndNonZero = nullptr;
	vecProblems = nullptr;
	viProblemState = nullptr;
}

globalCompass::~globalCompass() {
	if (vecProblems != nullptr) {
		for (int i = 0; i < nProblems; i++) {
			if (vecProblems[i] != nullptr) {
				delete vecProblems[i]; vecProblems[i] = nullptr;
			}
		}
		free(vecProblems); vecProblems = nullptr;
	}
	if (vSeed != nullptr) { free(vSeed); vSeed = nullptr; }
	if (vCountOnes != nullptr) { free(vCountOnes); vCountOnes = nullptr; }
	if (vSeedExtend != nullptr) { free(vSeedExtend); vSeedExtend = nullptr; }
	if (vIndNonZero != nullptr) { free(vIndNonZero); vIndNonZero = nullptr; }
	if (viProblemState != nullptr) { free(viProblemState); viProblemState = nullptr; }
}

int globalCompass::printUsage() {
	printf("Usage:\n");
	printf("1) Output folder/prefix\n");
	printf("2) Seed\n");
	printf("3) Start level (integer, optional)\n");
	printf("\n");
	return 0;
}

int binomial(int n, int k) {
	int iq;
	iq = 1;
	for (int i = n; i > n - k; i--) {
		iq *= i;
	}
	for (int i = k; i > 1; i--) {
		iq /= i;
	}
	return iq;
}

int globalCompass::findProblems() {
	int nTotalRows, nNonZeroRows, iL;
	int nW, cutLevel, iCount;
	int minLevel, iCurrentLevel;
	bool isSolutionFound;
	int* vecOrder;
	bool t;

	iCount = 0;

	nTotalRows = seedLen / 32;
	if (seedLen % 32 > 0) nTotalRows++;

	printf("Maximum number of 32-bit rows: %i\n", nTotalRows);

	vSeedExtend = (int*)malloc(sizeof(int) * 32 * nTotalRows);
	vCountOnes = (int*)malloc(sizeof(int) * nTotalRows);
	vIndNonZero = (int*)malloc(sizeof(int) * nTotalRows);

	for (int i = 0; i < seedLen; i++) {
		vSeedExtend[i] = vSeed[i];
	}
	for (int i = seedLen; i < 32 * nTotalRows; i++) {
		vSeedExtend[i] = 0;
	}

	nNonZeroRows = 0;
	printf("Number of 1-elements in each row\n");
	for (int k = 0; k < nTotalRows; k++) {
		vCountOnes[k] = 0;
		for (int i = 0; i < 32; i++) {
			vCountOnes[k] += vSeedExtend[32 * k + i];
		}
		printf("Row %i: %i\n", k + 1, vCountOnes[k]);
		if (vCountOnes[k] == 0) continue;
		vIndNonZero[nNonZeroRows] = k;
		nNonZeroRows++;
	}
	
	nW = seedWeight / 32;

	if (seedWeight % 32 == 0) {
		nProblems = binomial(nNonZeroRows, nW);
	}
	else {
		nProblems = nNonZeroRows * binomial(nNonZeroRows - 1, nW);
	}

	printf("Number of problems: %i\n", nProblems);

	vecProblems = (compSS**)malloc(sizeof(compSS*) * nProblems);

	for (int i = 0; i < nProblems; i++) {
		vecProblems[i] = new compSS(nTotalRows);
		sprintf(vecProblems[i]->outputFolderPrefix, "%s", outputFolderPrefix);
		vecProblems[i]->seedLenOrig = seedLen;
		vecProblems[i]->seedWeight = seedWeight;
		vecProblems[i]->nRows = nNonZeroRows;
		memcpy(vecProblems[i]->vSeedOrig, vSeedExtend, sizeof(int) * 32*nTotalRows);
	}
	
	vecOrder = (int*)malloc(sizeof(int) * nNonZeroRows);
	for (int i = 0; i < nNonZeroRows; i++) {
		vecOrder[i] = i;
	}

	if (seedWeight % 32 == 0) {
		cutLevel = nW - 1;
	}else {
		cutLevel = nW;
	}

	do {
		printf("cc: %i\t", iCount);
		for (int i = 0; i < nNonZeroRows; i++) {
			vecProblems[iCount]->vIndexRows[i] = vecOrder[i];
			printf("%i\t", vecOrder[i]);
		}
		printf("\n");
	
		iCount++;

		iL = cutLevel;

		do {
			if (iL == -1) break;
			vecOrder[iL]++;
			if (vecOrder[iL] == nNonZeroRows) {
				vecOrder[iL] = -1;
				iL--;
				continue;
			}
			t = false;
			for (int i = 0; i < iL; i++) {
				if (vecOrder[i] == vecOrder[iL]) {
					t = true;
					break;
				}
			}
			if (t) continue;

			if (iL < cutLevel) {
				iL++;
				continue;
			}

			for (int i = 0; i < nW - 1; i++) {
				if (vecOrder[i] > vecOrder[i + 1]) {
					t = true;
					break;
				}
			}
			if (!t) break;
		} while (true);

		if (iL == -1) break;
			
		for (int i = iL + 1; i < nNonZeroRows; i++) {
			for (int k = 0; k < nNonZeroRows; k++) {
				t = false;
				for (int j = 0; j < i; j++) {
					if (k == vecOrder[j]) {
						t = true;
						break;
					}
				}
				if (t) continue;
				vecOrder[i] = k;
				break;
			}
		}
	} while (true);

	printf("Count2: %i\n", iCount);

	free(vecOrder); vecOrder = nullptr;

	printf("\nNumber of problems: %i\n", nProblems);
	
	printf("\nInitiate problems:\n");

	int nV, nM, mL;
	for (int i = 0; i < nProblems; i++) {
		printf("Problem: %i\n", i);
		vecProblems[i]->setM();
		vecProblems[i]->prepareSeed();
		vecProblems[i]->prepareFileMiddleName();
		vecProblems[i]->findSeedLen();
		vecProblems[i]->findVariables();
		vecProblems[i]->formOriginalMatrix();
		vecProblems[i]->sortOriginalMatrix();
		nV = vecProblems[i]->nVar;
		nM = vecProblems[i]->vecMatrixSortedInfo[3];
		mL = nV / nM;
		if ((nV % nM) > 0) mL++;
		if (i == 0) {
			minLevel = mL;
		}else {
			if (mL < minLevel)minLevel = mL;
		}
	}

	printf("Min level: %i\n", minLevel);

	if (startLevel > minLevel) minLevel = startLevel;

	printf("Min level (corrected): %i\n", minLevel);
	
	isSolutionFound = false;

	viProblemState = (int*)malloc(sizeof(int) * nProblems);

	iCurrentLevel = minLevel - 1;

#pragma omp parallel
	{
		nThreads = omp_get_num_threads();
	}

	printf("\nNumber of threads: %i\n", nThreads);

	printf("\nStart processing:\n");

	do {
		iCurrentLevel++;
		for (int i = 0; i < nProblems; i++) {
			viProblemState[i] = 0;
		}
		printf("\nLevel %i\n", iCurrentLevel);
#pragma omp parallel
		{
			int ires, tid;
			bool isContinue;
			tid = omp_get_thread_num();
			for (int ip = 0; ip < nProblems; ip++) {
#pragma omp critical
				{
					isContinue = true;
					if (viProblemState[ip] == 1) {
						isContinue = false;
					}else {
						viProblemState[ip] = 1;
						printf("Thread #%i --- problem #%i\n", tid, ip);
					}
				}
				if (!isContinue)continue;
				ires = vecProblems[ip]->processLevel(iCurrentLevel);
				
#pragma omp critical
				{
					if (ires == 0) isSolutionFound = true;
				}
				
			}
		}
	} while (!isSolutionFound);

	return iCurrentLevel;
}


int globalCompass::startProcessing(int nA, char** vA) {
	int ires;
	if (!(nA == 3 || nA == 4)) {
		printf("Wrong number of arguments\n");
		printUsage();
		return -1;
	}

	if (nA == 4) {
		startLevel = atoi(vA[3]);
	}
	else {
		startLevel = MIN_START_LEVEL;
	}
	sprintf(outputFolderPrefix, "%s", vA[1]);
	printf("Output folder/prefix: \"%s\"\n", outputFolderPrefix);
	printf("Seed: \"%s\"\n", vA[2]);
	printf("Start level: %i\n", startLevel);
	seedLen = strlen(vA[2]);
	vSeed = (int*)malloc(sizeof(int) * (seedLen + 4));

	seedWeight = 0;
	for (int i = 0; i < seedLen; i++) {
		if (vA[2][i] == '1') {
			vSeed[i] = 1;
			seedWeight++;
		}
		else {
			vSeed[i] = 0;
		}
	}

	printf("Seed length: %i\n", seedLen);
	printf("Seed weight: %i\n", seedWeight);

	ires = findProblems();
	return ires;
}

compSS::compSS(int maxRows_in) {
	mPosGap = nullptr;
	mPosOne = nullptr;
	vPositionOne = nullptr;
	vPositionGap = nullptr;
	vIndGapReverse = nullptr;
	vIndOneReverse = nullptr;

	matrixOriginal = nullptr;
	matrixSorted = nullptr;

	mMatrixGapOriginal = nullptr;
	mMatrixOneOriginal = nullptr;
	mMatrixGapSorted = nullptr;
	mMatrixOneSorted = nullptr;
	mRowTemp = nullptr;
	vecMatrixSortedInfo = nullptr;
	vecMatrixOriginalInfo = nullptr;

	mRowFull = nullptr;
	mResultOne = nullptr;
	mResultGap = nullptr;

	vIndActiveState = nullptr;

	matColGap = nullptr;
	matColOne = nullptr;
	vIndexRowGap = nullptr;
	vIndexRowOne = nullptr;

	vCandidateRows = nullptr;

	matrixLocal = nullptr;
	mMatrixLocalOne = nullptr;
	mMatrixLocalGap = nullptr;

	mPreviousOne = nullptr;
	mPreviousGap = nullptr;


	maxRows = maxRows_in;
	
	vSeedLetter = (int*)malloc(sizeof(int) * maxRows * 32);
	vSeedBefore = (int*)malloc(sizeof(int) * maxRows * 32);
	vSeedOrig = (int*)malloc(sizeof(int) * maxRows * 32);
	vSeed = (int*)malloc(sizeof(int) * maxRows * 32);
	vIndexRows = (int*)malloc(sizeof(int) * maxRows);

	currentLevel = 0;

	isPrint = false;

	fo = nullptr;
		
	nVar = 0;
}

int compSS::freeMemory() {

	if (mPosGap != nullptr) { _aligned_free(mPosGap); mPosGap = nullptr; }
	if (mPosOne != nullptr) { _aligned_free(mPosOne); mPosOne = nullptr; }
	if (mMatrixOneOriginal != nullptr) { _aligned_free(mMatrixOneOriginal); mMatrixOneOriginal = nullptr; }
	if (mMatrixGapOriginal != nullptr) { _aligned_free(mMatrixGapOriginal); mMatrixGapOriginal = nullptr; }
	if (mMatrixOneSorted != nullptr) { _aligned_free(mMatrixOneSorted); mMatrixOneSorted = nullptr; }
	if (mMatrixGapSorted != nullptr) { _aligned_free(mMatrixGapSorted); mMatrixGapSorted = nullptr; }
	if (mRowTemp != nullptr) { _aligned_free(mRowTemp); mRowTemp = nullptr; }
	if (mRowFull != nullptr) { _aligned_free(mRowFull); mRowFull = nullptr; }
	if (mResultOne != nullptr) { _aligned_free(mResultOne); mResultOne = nullptr; }
	if (mResultGap != nullptr) { _aligned_free(mResultGap); mResultGap = nullptr; }
	if (mPreviousOne != nullptr) { _aligned_free(mPreviousOne); mPreviousOne = nullptr; }
	if (mPreviousGap != nullptr) { _aligned_free(mPreviousGap); mPreviousGap = nullptr; }

	

	if (vIndGapReverse != nullptr) { free(vIndGapReverse); vIndGapReverse = nullptr; }
	if (vIndOneReverse != nullptr) { free(vIndOneReverse); vIndOneReverse = nullptr; }
	if (vPositionGap != nullptr) { free(vPositionGap); vPositionGap = nullptr; }
	if (vPositionOne != nullptr) { free(vPositionOne); vPositionOne = nullptr; }
	if (matrixOriginal != nullptr) { free(matrixOriginal); matrixOriginal = nullptr; }
	if (matrixSorted != nullptr) { free(matrixSorted); matrixSorted = nullptr; }


	if (vIndActiveState != nullptr) { free(vIndActiveState); vIndActiveState = nullptr; }
		
	if (matColGap != nullptr) { free(matColGap); matColGap = nullptr; }
	if (matColOne != nullptr) { free(matColOne); matColOne = nullptr; }
	if (vIndexRowGap != nullptr) { free(vIndexRowGap); vIndexRowGap = nullptr; }
	if (vIndexRowOne != nullptr) { free(vIndexRowOne); vIndexRowOne = nullptr; }

	if (vecMatrixSortedInfo != nullptr) { free(vecMatrixSortedInfo); vecMatrixSortedInfo = nullptr; }
	if (vecMatrixOriginalInfo != nullptr) { free(vecMatrixOriginalInfo); vecMatrixOriginalInfo = nullptr; }

	if (vCandidateRows != nullptr) { free(vCandidateRows); vCandidateRows = nullptr; }

	if (matrixLocal != nullptr) { free(matrixLocal); matrixLocal = nullptr; }
	if (mMatrixLocalOne != nullptr) { _aligned_free(mMatrixLocalOne); mMatrixLocalOne = nullptr; }
	if (mMatrixLocalGap != nullptr) { _aligned_free(mMatrixLocalGap); mMatrixLocalGap = nullptr; }

	if (fo != nullptr) { 
		fprintf(fo, "\nElapsed time: %lli ms.\n\n", elapsedTime);
		fclose(fo); fo = nullptr; 
	}

	return 0;
}

compSS::~compSS() {
	if (vSeedLetter != nullptr) { free(vSeedLetter); vSeedLetter = nullptr; }
	if (vSeedBefore != nullptr) { free(vSeedBefore); vSeedBefore = nullptr; }
	if (vSeedOrig != nullptr) { free(vSeedOrig); vSeedOrig = nullptr;}
	if (vSeed != nullptr) { free(vSeed); vSeed = nullptr; }
	if (vIndexRows != nullptr) { free(vIndexRows); vIndexRows = nullptr;}
}

int compSS::prepareFileMiddleName() {
	int iValue;
	int* seedLoc;
	for (int i = 0; i < nRows; i++) {
		for (int j = 0; j < 8; j++) {
			seedLoc = vSeed + 32 * i + 4 * j;
			iValue = 8 * seedLoc[3] + 4 * seedLoc[2] + 2 * seedLoc[1] + seedLoc[0];
			sprintf(seedNameHex + 8 * i + j, "%1x", iValue);
		}
		
	}
	return 0;
}

int compSS::countBits(__m128i* vm) {
	long long n;
	n = 0;
	for (int i = 0; i < nVar128; i++) {
#ifdef WIN32
		n += __popcnt64(_mm_extract_epi64(vm[i], 0)) + __popcnt64(_mm_extract_epi64(vm[i], 1));
#else
		n += _mm_popcnt_u64(_mm_extract_epi64(vm[i], 0)) + _mm_popcnt_u64(_mm_extract_epi64(vm[i], 1));
#endif
	}
	return (int)n;
}

int compSS::printMatrixLocal(int iU) {
	int* matLoc;
	//__m128i* mLocGap, * mLocOne;
	//int aa[4];

	matLoc = matrixLocal + iU * nVar * iLevel;
	//mLocGap = mMatrixLocalGap + iU * nVar128 * iLevel;
	//mLocOne = mMatrixLocalOne + iU * nVar128 * iLevel;

	fprintf(fo, "\n");
	for (int i = 0; i < nVar; i++) {
		fprintf(fo, "%4i", i);
	}
	/*fprintf(fo, "\t");
	for (int i = 0; i < nVar; i++) {
		fprintf(fo, "%1i", i % 10);
	}
	fprintf(fo, "\t");
	for (int i = 0; i < nVar; i++) {
		fprintf(fo, "%1i", i % 10);
	}*/
	fprintf(fo, "\n");
	for (int i = 0; i < nVar; i++) {
		fprintf(fo, "----");
	}
	fprintf(fo, "\n");
	for (int i = 0; i < iLevel; i++) {
		for (int j = 0; j < nVar; j++) {
			if (matLoc[i * nVar + j] == -1) {
				fprintf(fo, "   _");
			}
			else {
				fprintf(fo, "%4i", matLoc[i * nVar + j]);
			}
		}
		/*fprintf(fo, "\t");
		for (int ii = 0; ii < nVar128; ii++) {
			_mm_store_si128((__m128i*)aa, mLocOne[i * nVar128 + ii]);
			for (int k = 0; k < 4; k++) {
				if (ii * 128 + 32 * k >= nVar)break;
				for (int j = 0; j < 32; j++) {
					if (ii * 128 + 32 * k + j >= nVar)break;
					fprintf(fo, "%1i", (1 & (aa[k] >> j)));
				}
			}
		}
		fprintf(fo, "\t");
		for (int ii = 0; ii < nVar128; ii++) {
			_mm_store_si128((__m128i*)aa, mLocGap[i * nVar128 + ii]);
			for (int k = 0; k < 4; k++) {
				if (ii * 128 + 32 * k >= nVar)break;
				for (int j = 0; j < 32; j++) {
					if (ii * 128 + 32 * k + j >= nVar)break;
					fprintf(fo, "%1i", (1 & (aa[k] >> j)));
				}
			}
		}*/
		fprintf(fo, "\n");
	}
	for (int i = 0; i < nVar; i++) {
		fprintf(fo, "----");
	}
	fprintf(fo, "\n");
	return 0;
}

int compSS::printMatrixOriginal() {
	int aa[4];
	fprintf(fo, "\n Ind | RoG RoO Sft Tot | ");
	for (int i = 0; i < nVar; i++) {
		fprintf(fo, "%4i", i);
	}
	fprintf(fo, "\tOne");
	for (int i = 0; i < nVar-3; i++) {
		fprintf(fo, " ");
	}
	fprintf(fo, "\tGap");
	for (int i = 0; i < nVar - 3; i++) {
		fprintf(fo, " ");
	}
	fprintf(fo, "\n");
	for (int i = 0; i < 4 * nVar + 25; i++) {
		fprintf(fo, "-");
	}
	fprintf(fo, "\n");
	for (int i = 0; i < nMatrixRows; i++) {
		fprintf(fo, "%4i | %3i %3i %3i %3i | ", i, vecMatrixOriginalInfo[i * SIZE_INFO + 0], vecMatrixOriginalInfo[i * SIZE_INFO + 1], vecMatrixOriginalInfo[i * SIZE_INFO + 2], vecMatrixOriginalInfo[i * SIZE_INFO + 3]);
		for (int j = 0; j < nVar; j++) {
			if (matrixOriginal[i * nVar + j] == -1) {
				fprintf(fo, "   _");
			}
			else {
				fprintf(fo, "%4i", matrixOriginal[i * nVar + j]);
			}
		}
		fprintf(fo, "\t");
		for (int ii = 0; ii < nVar128; ii++) {
			_mm_store_si128((__m128i*)aa, mMatrixOneOriginal[i * nVar128 + ii]);
			for (int k = 0; k < 4; k++) {
				if (ii * 128 + 32 * k >= nVar)break;
				for (int j = 0; j < 32; j++) {
					if (ii * 128 + 32 * k + j >= nVar)break;
					fprintf(fo, "%1i", (1 & (aa[k] >> j)));
				}
			}
		}
		fprintf(fo, "\t");
		for (int ii = 0; ii < nVar128; ii++) {
			_mm_store_si128((__m128i*)aa, mMatrixGapOriginal[i * nVar128 + ii]);
			for (int k = 0; k < 4; k++) {
				if (ii * 128 + 32 * k >= nVar)break;
				for (int j = 0; j < 32; j++) {
					if (ii * 128 + 32 * k + j >= nVar)break;
					fprintf(fo, "%1i", (1 & (aa[k] >> j)));
				}
			}
		}
		fprintf(fo, "\n");
	}
	for (int i = 0; i < 4 * nVar + 25; i++) {
		fprintf(fo, "-");
	}
	fprintf(fo, "\n");
	return 0;
}


int compSS::printMatrixSorted() {
	int aa[4];
	fprintf(fo, "\n Ind | RoG RoO Sft Tot | ");
	for (int i = 0; i < nVar; i++) {
		fprintf(fo, "%4i", i);
	}
	fprintf(fo, "\tOne");
	for (int i = 0; i < nVar - 3; i++) {
		fprintf(fo, " ");
	}
	fprintf(fo, "\tGap");
	for (int i = 0; i < nVar - 3; i++) {
		fprintf(fo, " ");
	}
	fprintf(fo, "\n");
	for (int i = 0; i < 4 * nVar + 25; i++) {
		fprintf(fo, "-");
	}
	fprintf(fo, "\n");

	for (int i = 0; i < nMatrixRows; i++) {
		fprintf(fo, "%4i | %3i %3i %3i %3i | ", i, vecMatrixSortedInfo[i * SIZE_INFO + 0], vecMatrixSortedInfo[i * SIZE_INFO + 1], vecMatrixSortedInfo[i * SIZE_INFO + 2], vecMatrixSortedInfo[i * SIZE_INFO + 3]);
		for (int j = 0; j < nVar; j++) {
			if (matrixSorted[i * nVar + j] == -1) {
				fprintf(fo, "   _");
			}
			else {
				fprintf(fo, "%4i", matrixSorted[i * nVar + j]);
			}
		}
		fprintf(fo, "\t");
		for (int ii = 0; ii < nVar128; ii++) {
			_mm_store_si128((__m128i*)aa, mMatrixOneSorted[i * nVar128 + ii]);
			for (int k = 0; k < 4; k++) {
				if (ii * 128 + 32 * k >= nVar)break;
				for (int j = 0; j < 32; j++) {
					if (ii * 128 + 32 * k + j >= nVar)break;
					fprintf(fo, "%1i", (1 & (aa[k] >> j)));
				}
			}
		}
		fprintf(fo, "\t");
		for (int ii = 0; ii < nVar128; ii++) {
			_mm_store_si128((__m128i*)aa, mMatrixGapSorted[i * nVar128 + ii]);
			for (int k = 0; k < 4; k++) {
				if (ii * 128 + 32 * k >= nVar)break;
				for (int j = 0; j < 32; j++) {
					if (ii * 128 + 32 * k + j >= nVar)break;
					fprintf(fo, "%1i", (1 & (aa[k] >> j)));
				}
			}
		}
		fprintf(fo, "\n");
	}
	for (int i = 0; i < 4 * nVar + 25; i++) {
		fprintf(fo, "-");
	}
	fprintf(fo, "\n");
	return 0;
}

int compSS::printCandidateMatrix(int n) {
	int indR;

	fprintf(fo, "\n # |  Ind | RoG RoO Sft Tot | ");
	for (int i = 0; i < nVar; i++) {
		fprintf(fo, "%4i", i);
	}
	fprintf(fo, "\n");

	for (int i = 0; i < 4*nVar + 30; i++) {
		fprintf(fo, "-");
	}
	fprintf(fo, "\n");

	for (int i = 0; i < n; i++) {
		indR = vCandidateRows[i];
		fprintf(fo, "%2i | %4i | %3i %3i %3i %3i | ", i, indR, vecMatrixSortedInfo[indR * SIZE_INFO + 0], vecMatrixSortedInfo[indR * SIZE_INFO + 1], vecMatrixSortedInfo[indR * SIZE_INFO + 2], vecMatrixSortedInfo[indR * SIZE_INFO + 3]);
		for (int j = 0; j < nVar; j++) {
			if (matrixSorted[indR * nVar + j] == -1) {
				fprintf(fo, "   _");
			}
			else {
				fprintf(fo, "%4i", matrixSorted[indR * nVar + j]);
			}
		}
		fprintf(fo, "\n");
	}
	for (int i = 0; i < 4 * nVar + 30; i++) {
		fprintf(fo, "-");
	}
	fprintf(fo, "\n");
	return 0;
}


int compSS::printSeed() {
	int ijk;
	
	fprintf(fo, "Seed\n");
	for (int i = 0; i < seedLen; i++) {
		fprintf(fo, "%1i", vSeed[i]);
	}
	fprintf(fo, "\n\n");
	fprintf(fo, "Length: %i\n\n", seedLen);
	fprintf(fo, "Number of rows: %i\n\n", n32);
	for (int i = 0; i < n32; i++) {
		for (int j = 0; j < 4; j++) {
			for (int k = 0; k < 8; k++) {
				ijk = 32 * i + 8 * j + k;
				if (ijk >= seedLen) {
					fprintf(fo, "0");
				}
				else {
					fprintf(fo, "%1i", vSeed[ijk]);
				}
			}
			if (j < 3)fprintf(fo, " ");
		}
		fprintf(fo, "\n");
	}
	fprintf(fo, "\n");
	return 0;
}

int compSS::findNonZeroElement(__m128i* vm) {
	int iBase;
	short iVal;

	iBase = 0;
	for (int ir = 0; ir < nVar128; ir++) {
		for (int j = 0; j < 8; j++) {
			iVal = *((short *)(vm + ir) + j);
			if (iVal == 0) {
				iBase += 16;
				if (iBase >= nVar) return -1;
				continue;
			}
			for (int k = 0; k < 16; k++) {
				if ((1 & (iVal >> k)) == 0)continue;
				return (iBase + k);
			}
		}
		
	}
	return -1;
}

int compSS::printVector128(__m128i* vm) {
	int AA[4];
	for (int i = 0; i < nVar; i++) {
		fprintf(fo, "%1i", i % 10);
		if ((i % 10) == 9) fprintf(fo, " ");
	}
	fprintf(fo, "\n");

	for (int i = 0; i < nVar128; i++) {
		_mm_store_si128((__m128i*)AA, vm[i]);
		for (int k = 0; k < 4; k++) {
			if (128 * i + 32 * k >= nVar) break;
			for (int j = 0; j < 32; j++) {
				if (128 * i + 32 * k + j>= nVar) break;
				fprintf(fo, "%1i", (AA[k] >> j) & 1);
				if ((128 * i + 32 * k + j) % 10 == 9) fprintf(fo, " ");
			}
		}
	}
	fprintf(fo, "\n");
	return 0;
}

int compSS::processLevel(int iLevel_in) {
	int iL, ires;
	int iRow, sizeLeftOne, sizeLeftGap, maxSizeLeft, maxChunk, indMinRow;
	bool isFound;

	isFound = false;

	iLevel = iLevel_in;

	for (int i = 0; i < nVar128; i++) {
		mResultOne[i] = mRowFull[i];
		mResultGap[i] = mRowFull[i];
	}

	for (int i = 0; i < nVar; i++) {
		vIndexRowGap[i] = 0;
		vIndexRowOne[i] = 0;
	}

	iCount = 0;

	vIndexRowOne[0] = -1;

	iL = 0;
	indMinRow = 0;
	vCandidateRows[iL] = -1;

	maxSizeLeft = nVar;
	do {
		vCandidateRows[iL]++;
		if (vCandidateRows[iL] == nMatrixRows - (iLevel - iL)) {
			iL--;
			if (iL < 0)break;
			continue;
		}
		
		indMinRow = vCandidateRows[iL];
		maxChunk = vecMatrixSortedInfo[indMinRow * SIZE_INFO + 3];

		if (maxSizeLeft > (iLevel - iL) * maxChunk) {
			iL--;
			if (iL < 0) break;
			continue;
		}

		iRow = vCandidateRows[iL];

		for (int i = 0; i < nVar128; i++) {
			mResultOne[(iL + 1) * nVar128 + i] = _mm_andnot_si128(mMatrixOneSorted[iRow * nVar128 + i], mResultOne[iL * nVar128 + i]);
		}
		for (int i = 0; i < nVar128; i++) {
			mResultGap[(iL + 1) * nVar128 + i] = _mm_andnot_si128(mMatrixGapSorted[iRow * nVar128 + i], mResultGap[iL * nVar128 + i]);
		}

		sizeLeftOne = countBits(mResultOne + (iL + 1) * nVar128);
		sizeLeftGap = countBits(mResultGap + (iL + 1) * nVar128);
		maxSizeLeft = sizeLeftOne;
		if (maxSizeLeft < sizeLeftGap) maxSizeLeft = sizeLeftGap;

		if (iL + 1 < iLevel) {
			iL++;
			vCandidateRows[iL] = vCandidateRows[iL - 1];
			continue;
		}

		if (maxSizeLeft > 0) continue;

		initiateLocalMatrix();
		ires = processLocalMatrix(0);
		if (ires == 0) isFound = true;
	} while (true);

	if (isFound) return 0;
	return -1;
}

int compSS::findOneIn128(__m128i mA, int iStart) {
	int AA[4];
	int ii1, ii2, k;
	if (iStart >= 128) return -1;
	ii1 = iStart >> 5;
	ii2 = iStart & 31;
	_mm_store_si128((__m128i*)AA, mA);

	for (int j = ii2; j < 32; j++) {
		k = 32 * ii1 + j;
		if ((1 & (AA[ii1] >> j)) == 1) return k;
	}

	for (int i = ii1 + 1; i < 4; i++) {
		for (int j = 0; j < 32; j++) {
			k = 32 * i + j;
			if ((1 & (AA[i] >> j)) == 1) return k;
		}
	}
	return -1;
}

bool checkMzero(__m128i mA) {
	if (_mm_extract_epi64(mA, 0) != 0) return false;
	if (_mm_extract_epi64(mA, 1) != 0) return false;
	return true;
}

int compSS::openNewFile() {
	sprintf(outputFileName, "%s_Level_%i_seed_%s.txt", outputFolderPrefix, iLevel, seedNameHex);
	fo = fopen(outputFileName, "w");
	if (fo == nullptr) {
		printf("Error: cannot open file %s\n", outputFileName);
		return -1;
	}

	fprintf(fo, "Original seed\n");
	for (int i = 0; i < seedLenOrig; i++) {
		fprintf(fo, "%1i", vSeedOrig[i]);
	}
	fprintf(fo, "\n\n");

	int n;

	n = seedLenOrig >> 5;
	if ((seedLenOrig & 31) > 0) n++;

	for (int i = 0; i < seedLenOrig; i++) {
		fprintf(fo, "%1i", vSeedOrig[i]);
		if (i % 32 == 31)fprintf(fo, "\n");
	}
	fprintf(fo, "\n\n");

	fprintf(fo, "Length (original seed): %i\n\n", seedLenOrig);
	fprintf(fo, "Weight: %i\n\n", seedWeight);
	fprintf(fo, "Total number of rows: %i\n\n", n);

	fprintf(fo, "Number of rows: %i\n\n", nRows);

	fprintf(fo, "Indices of rows (new: original):\n");
	for (int i = 0; i < nRows; i++) {
		fprintf(fo, "%i: %i\n", i, vIndexRows[i]);
	}
	fprintf(fo, "\n");

	for (int i = 0; i < nRows; i++) {
		for (int j = 0; j < 32; j++) {
			fprintf(fo, "%1i", vSeed[32 * i + j]);
		}
		fprintf(fo, "\n");
	}
	fprintf(fo, "\n");

	fprintf(fo, "Number of variables (gaps/ones): %i\n\n", nVar);

	fprintf(fo, "Positions of gaps:\n");
	for (int i = 0; i < nVar; i++) {
		fprintf(fo, "%i: %i\n", i + 1, vPositionGap[i]);
	}

	fprintf(fo, "\nPositions of ones:\n");
	for (int i = 0; i < nVar; i++) {
		fprintf(fo, "%i: %i\n", i + 1, vPositionOne[i]);
	}

	fprintf(fo, "\n");

	fprintf(fo, "Original matrix\n\n");

	printMatrixOriginal();

	fprintf(fo, "Sorted matrix\n\n");

	printMatrixSorted();

	return 0;
}

int compSS::printSolutionLetters(int iU) {
	char CC, CC2;
	int *matLoc, iVal, nOut, indR, iRowO, iShift;

	nOut = (seedWeight >> 5);
	if ((seedWeight & 31) > 0) nOut++;

	seedLetterAfter[seedWeight] = '\0';

	matLoc = matrixLocal + iU * nVar * iLevel;

	for (int i = 0; i < seedLen; i++) {
		seedLetterBefore[i] = '0';
		if(vSeed[i] == 1) seedLetterBefore[i] = '1';
	}

	for (int i = 0; i < seedWeight; i++) {
		if (seedLetterBefore[i] == '1') {
			CC = 'A' + (i >> 5);
			seedLetterBefore[i] = CC;
			seedLetterAfter[i] = CC;
		}
		else {
			seedLetterAfter[i] = '0';
		}
	}

	for (int iL = 0; iL < iLevel; iL++) {
		CC2 = CC + iL + 1;
		for (int j = 0; j < nVar; j++) {
			iVal = matLoc[iL * nVar + j];
			if (iVal == -1) continue;
			seedLetterBefore[vPositionOne[iVal]] = CC2;
			seedLetterAfter[vPositionGap[j]] = CC2;
		}
	}

	fprintf(fo, "\nSeed (before)\n\n");

	for (int i = 0; i < seedLen; i++) {
		fprintf(fo, "%1c", seedLetterBefore[i]);
		if ((i & 31) == 31) fprintf(fo, "\n");
	}
	fprintf(fo, "\n");

	fprintf(fo, "Seed (after)\n\n");

	for (int i = 0; i < seedWeight; i++) {
		fprintf(fo, "%1c", seedLetterAfter[i]);
		if ((i & 31) == 31) fprintf(fo, "\n");
	}
	fprintf(fo, "\n");

	fprintf(fo, "\nProcessing scheme\n\n");

	for (int i = 0; i < nOut; i++) {
		fprintf(fo, "Output row #%i\n", i);
		fprintf(fo, "Input row #%3i", i);
		for (int j = 0; j < 31; j++) {
			fprintf(fo, " ");
		}
		for (int k = 0; k < 32; k++) {
			if (seedLetterBefore[32 * i + k] == ('A' + i)) {
				fprintf(fo, "%1c", seedLetterBefore[32 * i + k]);
			}else {
				fprintf(fo, "0");
			}
		}
		fprintf(fo, "\n");
		for (int iL = 0; iL < iLevel; iL++) {
			indR = vCandidateRows[iL];
			if (vecMatrixSortedInfo[indR * SIZE_INFO + 0] != i)continue;
			iRowO = vecMatrixSortedInfo[indR * SIZE_INFO + 1];
			iShift = vecMatrixSortedInfo[indR * SIZE_INFO + 2];
			fprintf(fo, "Input row #%3i", iRowO);
			for (int j = 0; j < 31 - iShift; j++) {
				fprintf(fo, " ");
			}
			for (int k = 0; k < 32; k++) {
				if (seedLetterBefore[32 * iRowO + k] == CC + iL + 1) {
					fprintf(fo, "%1c", seedLetterBefore[32 * iRowO + k]);
				}
				else {
					fprintf(fo, "0");
				}
			}
			fprintf(fo, "\n");
		}
		
		fprintf(fo, "\n");
	}

	return 0;
}


int compSS::printSIMDsolution(int iU) {
	int* matLoc, iVal, nOut, indR, iRowO, iShift;
	int u;

	matLoc = matrixLocal + iU * nVar * iLevel;

	nOut = (seedWeight >> 5);
	if ((seedWeight & 31) > 0) nOut++;

	fprintf(fo, "SIMD function\n\n");

	fprintf(fo, "void compactSeed(__m128i *vmIn, __m128i *vmOut){\n");

	for (int i = 0; i < nOut; i++) {
		u = 0;
		for (int k = 0; k < 32; k++) {
			if (32 * i + k >= seedWeight) break;
			if (vSeed[32 * i + k] == 0) continue;
			u |= (1 << k);
		}
		fprintf(fo, "\tvmOut[%i] = _mm_and_si128(vmIn[%i], _mm_set1_epi32(0x%08x));\n", i, vIndexRows[i], u);
		
		for (int iL = 0; iL < iLevel; iL++) {
			indR = vCandidateRows[iL];
			if (vecMatrixSortedInfo[indR * SIZE_INFO + 0] != i)continue;
			iRowO = vecMatrixSortedInfo[indR * SIZE_INFO + 1];
			iShift = vecMatrixSortedInfo[indR * SIZE_INFO + 2];

			fprintf(fo, "\tvmOut[%i] = _mm_or_si128(vmOut[%i], ", i, i);
			if(iShift < 0){
				fprintf(fo, "_mm_slli_epi32(");
			}else if (iShift > 0) {
				fprintf(fo, "_mm_srli_epi32(");
			}
			u = 0;
			for (int k = 0; k < nVar; k++) {
				iVal = matLoc[iL * nVar + k];
				if (iVal == -1) continue;
				u |= (1 << (31 & vPositionOne[iVal]));
			}

			fprintf(fo, "_mm_and_si128(vmIn[%i], _mm_set1_epi32(0x%08x))", vIndexRows[iRowO], u);

			if (iShift < 0) {
				fprintf(fo, ", %i)", -iShift);
			}else if (iShift > 0) {
				fprintf(fo, ", %i)", iShift);
			}

			fprintf(fo, ");\n");
			
		}

		
	}

	fprintf(fo, "}\n\n");

	return 0;
}



int compSS::processLocalMatrix(int iU) {
	__m128i* mLocOne, *mLocGap;
	__m128i mA, mB, mC, mD, mE;
	int* matLoc;
	int countE, countEMin, indCol;
	int indOut, iStart, iElement, iCol, iElement2;
	int iE1, iE2, iC1, iC2;
	int iRstart, oState, oStateLoc;
	bool tt;
	
	mLocGap = mMatrixLocalGap + iU * iLevel * nVar128;
	mLocOne = mMatrixLocalOne + iU * iLevel * nVar128;
	matLoc = matrixLocal + iU * nVar * iLevel;

	for (int i = 0; i < nVar128; i++) {
		mPreviousGap[i] = mZero;
		mPreviousOne[i] = mZero;
	}

	do{
		tt = false;

		//Gap
		for (int i = 0; i < nVar128; i++) {
			mA = mZero;
			mB = mZero;
			for (int j = 0; j < iLevel; j++) {
				mB = _mm_or_si128(mB, _mm_and_si128(mA, mLocGap[j * nVar128 + i]));
				mA = _mm_or_si128(mA, mLocGap[j * nVar128 + i]);
			}

			mD = _mm_and_si128(mRowFull[i], mA);
			mE = _mm_xor_si128(mRowFull[i], mD);
			if (!checkMzero(mE)) return -1;

			mC = _mm_xor_si128(mRowFull[i], mB);
			mA = _mm_andnot_si128(mPreviousGap[i], mC);

			if (checkMzero(mA)) continue;
			tt = true;

			mPreviousGap[i] = _mm_or_si128(mPreviousGap[i], mC);

			for (int j = 0; j < iLevel; j++) {
				iStart = 0;
				mB = _mm_and_si128(mA, mLocGap[j * nVar128 + i]);
				do {
					indOut = findOneIn128(mB, iStart);
					if (indOut < 0) break;
					iElement = matLoc[j * nVar + i * 128 + indOut];
					iE1 = iElement >> 7;
					iE2 = iElement & 127;
					for (int jj = 0; jj < iLevel; jj++) {
						if (j == jj)continue;
						mE = mLocOne[jj * nVar128 + iE1];
						mD = _mm_and_si128(mOne[iE2], mE);
						if (!checkMzero(mD)) {
							mLocOne[jj * nVar128 + iE1] = _mm_xor_si128(mLocOne[jj * nVar128 + iE1], mOne[iE2]);
							for (int k = 0; k < nVar; k++) {
								if (matLoc[jj * nVar + k] == iElement) {
									mLocGap[jj * nVar128 + (k >> 7)] = _mm_xor_si128(mLocGap[jj * nVar128 + (k >> 7)], mOne[k & 127]);
									matLoc[jj * nVar + k] = -1;
									break;
								}
							}
						}
					}

					iStart = indOut + 1;
				} while (true);
			}
		}

		//One
		for (int i = 0; i < nVar128; i++) {
			mA = mZero;
			mB = mZero;
			for (int j = 0; j < iLevel; j++) {
				mB = _mm_or_si128(mB, _mm_and_si128(mA, mLocOne[j * nVar128 + i]));
				mA = _mm_or_si128(mA, mLocOne[j * nVar128 + i]);
			}

			mD = _mm_and_si128(mRowFull[i], mA);
			mE = _mm_xor_si128(mRowFull[i], mD);
			if (!checkMzero(mE)) return -2;

			mC = _mm_xor_si128(mRowFull[i], mB);
			mA = _mm_andnot_si128(mPreviousOne[i], mC);

			if (checkMzero(mA)) continue;
			tt = true;

			mPreviousOne[i] = _mm_or_si128(mPreviousOne[i], mC);

			for (int j = 0; j < iLevel; j++) {
				iStart = 0;
				mB = _mm_and_si128(mA, mLocOne[j * nVar128 + i]);
				do {
					indOut = findOneIn128(mB, iStart);
					if (indOut < 0) break;
					iElement = i * 128 + indOut;

					for (int k = 0; k < nVar; k++) {
						if (matLoc[j * nVar + k] == iElement) {
							iCol = k;
							break;
						}
					}

					iC1 = iCol >> 7;
					iC2 = iCol & 127;

					for (int jj = 0; jj < iLevel; jj++) {
						if (j == jj) continue;
						iElement2 = matLoc[jj * nVar + iCol];
						if (iElement2 < 0) continue;
						iE1 = iElement2 >> 7;
						iE2 = iElement2 & 127;
						matLoc[jj * nVar + iCol] = -1;
						mLocGap[jj * nVar128 + iC1] = _mm_xor_si128(mLocGap[jj * nVar128 + iC1], mOne[iC2]);
						mLocOne[jj * nVar128 + iE1] = _mm_xor_si128(mLocOne[jj * nVar128 + iE1], mOne[iE2]);
					}

					iStart = indOut + 1;
				} while (true);
			}
		}

	}while (tt);
	
	countEMin = 0;
	indCol = 0;
	for (int i = 0; i < iLevel; i++) {
		if (matLoc[i * nVar + 0] > -1) countEMin++;
	}

	for (int j = 1; j < nVar; j++) {
		countE = 0;
		for (int i = 0; i < iLevel; i++) {
			if (matLoc[i * nVar + j] < 0)continue;
			countE++;
		}
		if (countE < 2) continue;
		if (countEMin == 1) {
			countEMin = countE;
			indCol = j;
		}else if (countE < countEMin) {
			countEMin = countE;
			indCol = j;
		}
	}
	
	if (countEMin == 1) {
		if (iCount == 0) openNewFile();
		iCount++;
		fprintf(fo, "\n===========================================================================\n\n");
		fprintf(fo, "Solution: %lli\n", iCount);
		fprintf(fo, "\nOriginal submatrix\n");
		printCandidateMatrix(iLevel);
		fprintf(fo, "\nSubmatrix after correction\n");
		printMatrixLocal(iU);
		printSolutionLetters(iU);
		printSIMDsolution(iU);
		return 0;
	}

	mLocGap = mMatrixLocalGap + (iU + 1) * iLevel * nVar128;
	mLocOne = mMatrixLocalOne + (iU + 1) * iLevel * nVar128;
	matLoc = matrixLocal + (iU + 1) * nVar * iLevel;

	oState = -3;
	iRstart = 0;

	iC1 = indCol >> 7;
	iC2 = indCol & 127;

	for (int ic = 0; ic < countEMin; ic++) {
		memcpy(mMatrixLocalGap + (iU + 1) * iLevel * nVar128, mMatrixLocalGap + iU * iLevel * nVar128, iLevel* nVar128 * sizeof(__m128i));
		memcpy(mMatrixLocalOne + (iU + 1) * iLevel * nVar128, mMatrixLocalOne + iU * iLevel * nVar128, iLevel* nVar128 * sizeof(__m128i));
		memcpy(matrixLocal + (iU + 1) * iLevel * nVar, matrixLocal + iU * iLevel * nVar, iLevel* nVar * sizeof(int));

		for (int i = iRstart; i < iLevel; i++) {
			if (matLoc[i * nVar + indCol] < 0) continue;
			iRstart = i;
			break;
		}

		for (int i = 0; i < iLevel; i++) {
			if (i == iRstart)continue;
			iElement = matLoc[i * nVar + indCol];
			if (iElement < 0) continue;

			iE1 = iElement >> 7;
			iE2 = iElement & 127;

			mLocGap[i * nVar128 + iC1] = _mm_xor_si128(mLocGap[i * nVar128 + iC1], mOne[iC2]);
			mLocOne[i * nVar128 + iE1] = _mm_xor_si128(mLocOne[i * nVar128 + iE1], mOne[iE2]);
			matLoc[i * nVar + indCol] = -1;
		}

		oStateLoc = processLocalMatrix(iU + 1);
		if (oStateLoc > oState) oState = oStateLoc;

		iRstart++;
	}

	return oState;
}

int compSS::initiateLocalMatrix() {
	for (int i = 0; i < iLevel; i++) {
		for (int j = 0; j < nVar; j++) {
			matrixLocal[i * nVar + j] = matrixSorted[vCandidateRows[i] * nVar + j];
		}
	}

	for (int i = 0; i < iLevel; i++) {
		for (int j = 0; j < nVar128; j++) {
			mMatrixLocalGap[i * nVar128 + j] = mMatrixGapSorted[vCandidateRows[i] * nVar128 + j];
		}
	}

	for (int i = 0; i < iLevel; i++) {
		for (int j = 0; j < nVar128; j++) {
			mMatrixLocalOne[i * nVar128 + j] = mMatrixOneSorted[vCandidateRows[i] * nVar128 + j];
		}
	}
	return 0;
}

int compSS::formOriginalMatrix() {
	__m128i mA, mB, mC, mD;
	int ic, ii1, ii2;
	int indG, indO, iExtract, nCount;

	nMatrixRows = 0;

	for (int iGap = 0; iGap < nRows; iGap++) {
		if (_mm_extract_epi32(mPosGap[iGap], 0) == 0) continue;
		mA = mPosGap[iGap];
		for (int iOne = 0; iOne < nRows; iOne++) {
			if (_mm_extract_epi32(mPosOne[iOne], 0) == 0) continue;
			mB = mPosOne[iOne];
			for (int i = 0; i < 32; i++) {
				mC = _mm_srli_epi32(mB, i);
				mD = _mm_and_si128(mA, mC);
				if (_mm_extract_epi32(mD, 0) != 0) nMatrixRows++;
			}
			for (int i = 1; i < 32; i++) {
				mC = _mm_slli_epi32(mB, i);
				mD = _mm_and_si128(mA, mC);
				if (_mm_extract_epi32(mD, 0) != 0) nMatrixRows++;
			}
		}
	}

	mPreviousOne = (__m128i*)_aligned_malloc(sizeof(__m128i) * nVar128, sizeof(__m128i));
	mPreviousGap = (__m128i*)_aligned_malloc(sizeof(__m128i) * nVar128, sizeof(__m128i));
	mMatrixOneOriginal = (__m128i*)_aligned_malloc(sizeof(__m128i) * nVar128 * nMatrixRows, sizeof(__m128i));
	mMatrixGapOriginal = (__m128i*)_aligned_malloc(sizeof(__m128i) * nVar128 * nMatrixRows, sizeof(__m128i));
	mMatrixOneSorted = (__m128i*)_aligned_malloc(sizeof(__m128i) * nVar128 * nMatrixRows, sizeof(__m128i));
	mMatrixGapSorted = (__m128i*)_aligned_malloc(sizeof(__m128i) * nVar128 * nMatrixRows, sizeof(__m128i));
	mRowTemp = (__m128i*)_aligned_malloc(sizeof(__m128i) * nVar128, sizeof(__m128i));
	mRowFull = (__m128i*)_aligned_malloc(sizeof(__m128i) * nVar128, sizeof(__m128i));
	mResultOne = (__m128i*)_aligned_malloc(sizeof(__m128i) * nVar128 * (nVar + 1), sizeof(__m128i));
	mResultGap = (__m128i*)_aligned_malloc(sizeof(__m128i) * nVar128 * (nVar + 1), sizeof(__m128i));
	matrixOriginal = (int*)malloc(sizeof(int) * nMatrixRows * nVar);
	matrixSorted = (int*)malloc(sizeof(int) * nMatrixRows * nVar);
	vecMatrixSortedInfo = (int*)malloc(sizeof(int) * nMatrixRows * SIZE_INFO);
	vecMatrixOriginalInfo = (int*)malloc(sizeof(int) * nMatrixRows * SIZE_INFO);
	matColGap = (int*)malloc(sizeof(int) * nVar * nVar);
	matColOne = (int*)malloc(sizeof(int) * nVar * nVar);
	vIndexRowGap = (int*)malloc(sizeof(int) * nVar);
	vIndexRowOne = (int*)malloc(sizeof(int) * nVar);
	vIndActiveState = (int*)malloc(sizeof(int) * nVar);
	vCandidateRows = (int*)malloc(sizeof(int) * nVar);

	matrixLocal = (int*)malloc(sizeof(int) * nVar * MAX_ROWS * nVar);
	mMatrixLocalOne = (__m128i*)_aligned_malloc(sizeof(__m128i) * nVar * MAX_ROWS * nVar128, sizeof(__m128i));
	mMatrixLocalGap = (__m128i*)_aligned_malloc(sizeof(__m128i) * nVar * MAX_ROWS * nVar128, sizeof(__m128i));

	for (int i = 0; i < nVar128; i++) {
		mResultOne[i] = mZero;
		mResultGap[i] = mZero;
		mRowFull[i] = mZero;
	}

	for (int i = 0; i < nVar; i++) {
		ii1 = i >> 7;
		ii2 = i & 127;
		mRowFull[ii1] = _mm_or_si128(mRowFull[ii1], mOne[ii2]);
	}

	for (int i = 0; i < nMatrixRows * nVar; i++) {
		matrixOriginal[i] = -1;
	}

	for (int i = 0; i < nVar128 * nMatrixRows; i++) {
		mMatrixOneOriginal[i] = mZero;
	}
	for (int i = 0; i < nVar128 * nMatrixRows; i++) {
		mMatrixGapOriginal[i] = mZero;
	}

	ic = 0;

	for (int i = 0; i < nVar; i++) {
		vIndexRowGap[i] = 0;
		vIndexRowOne[i] = 0;
	}

	for (int iGap = 0; iGap < nRows; iGap++) {
		if (_mm_extract_epi32(mPosGap[iGap], 0) == 0) continue;
		mA = mPosGap[iGap];
		for (int iOne = 0; iOne < nRows; iOne++) {
			if (_mm_extract_epi32(mPosOne[iOne], 0) == 0) continue;
			mB = mPosOne[iOne];
			for (int i = 31; i > -32; i--) {
				if (i < 0) {
					mC = _mm_slli_epi32(mB, -i);
				}else {
					mC = _mm_srli_epi32(mB, i);
				}
				mD = _mm_and_si128(mA, mC);
				iExtract = _mm_extract_epi32(mD, 0);
				if (iExtract != 0){
					nCount = 0;
					for (int k = 0; k < 32; k++) {
						if ((1 & (iExtract >> k)) == 0) continue;
						indG = vIndGapReverse[32 * iGap + k];
						indO = vIndOneReverse[32 * iOne + k + i];
						matColOne[indO * nVar + vIndexRowOne[indO]] = ic;
						vIndexRowOne[indO]++;
						matColGap[indG * nVar + vIndexRowGap[indG]] = ic;
						vIndexRowGap[indG]++;
						matrixOriginal[ic * nVar + indG] = indO;
						ii1 = indO >> 7;
						ii2 = indO & 127;
						mMatrixOneOriginal[ic * nVar128 + ii1] = _mm_or_si128(mMatrixOneOriginal[ic * nVar128 + ii1], mOne[ii2]);
						ii1 = indG >> 7;
						ii2 = indG & 127;
						mMatrixGapOriginal[ic * nVar128 + ii1] = _mm_or_si128(mMatrixGapOriginal[ic * nVar128 + ii1], mOne[ii2]);
						nCount++;
					}
					vecMatrixOriginalInfo[ic * SIZE_INFO + 0] = iGap;
					vecMatrixOriginalInfo[ic * SIZE_INFO + 1] = iOne;
					vecMatrixOriginalInfo[ic * SIZE_INFO + 2] = i;
					vecMatrixOriginalInfo[ic * SIZE_INFO + 3] = nCount;
					ic++;
				}
			}
		}
	}

	maxElementsInRow = 0;
	for (int i = 0; i < nMatrixRows; i++) {
		if (vecMatrixOriginalInfo[SIZE_INFO * i + 3] > maxElementsInRow) maxElementsInRow = vecMatrixOriginalInfo[SIZE_INFO * i + 3];
	}
	
	return 0;
}


int compSS::sortOriginalMatrix() {
	__m128i mA;
	int iVal, indG, indO, iL;

	iL = 0;

	for (int i = 0; i < nMatrixRows*nVar; i++) {
		matrixSorted[i] = matrixOriginal[i];
	}

	for (int i = 0; i < nMatrixRows * nVar128; i++) {
		mMatrixOneSorted[i] = mMatrixOneOriginal[i];
		mMatrixGapSorted[i] = mMatrixGapOriginal[i];
	}

	for (int i = 0; i < 4 * nMatrixRows; i++) {
		vecMatrixSortedInfo[i] = vecMatrixOriginalInfo[i];
	}

	for (int m = maxElementsInRow; m > 1; m--) {
		for (int k = iL; k < nMatrixRows; k++) {
			if (vecMatrixSortedInfo[k * SIZE_INFO + 3] < m) continue;
			for (int i = 0; i < SIZE_INFO; i++) {
				iVal = vecMatrixSortedInfo[k * SIZE_INFO + i];
				vecMatrixSortedInfo[k * SIZE_INFO + i] = vecMatrixSortedInfo[iL * SIZE_INFO + i];
				vecMatrixSortedInfo[iL * SIZE_INFO + i] = iVal;
			}

			for (int i = 0; i < nVar; i++) {
				iVal = matrixSorted[k * nVar + i];
				matrixSorted[k * nVar + i] = matrixSorted[iL * nVar + i];
				matrixSorted[iL * nVar + i] = iVal;
			}
			for (int i = 0; i < nVar128; i++) {
				mA = mMatrixOneSorted[k * nVar128 + i];
				mMatrixOneSorted[k * nVar128 + i] = mMatrixOneSorted[iL * nVar128 + i];
				mMatrixOneSorted[iL * nVar128 + i] = mA;
			}
			for (int i = 0; i < nVar128; i++) {
				mA = mMatrixGapSorted[k * nVar128 + i];
				mMatrixGapSorted[k * nVar128 + i] = mMatrixGapSorted[iL * nVar128 + i];
				mMatrixGapSorted[iL * nVar128 + i] = mA;
			}
			iL++;
		}
	}

	for (int i = 0; i < nVar; i++) {
		vIndexRowGap[i] = 0;
		vIndexRowOne[i] = 0;
	}
	
	for (int i = 0; i < nMatrixRows; i++) {
		for (int j = 0; j < nVar; j++) {
			if (matrixSorted[i * nVar + j] == -1)continue;
			indO = matrixSorted[i * nVar + j];
			indG = j;

			matColOne[indO * nVar + vIndexRowOne[indO]] = i;
			vIndexRowOne[indO]++;
			matColGap[indG * nVar + vIndexRowGap[indG]] = i;
			vIndexRowGap[indG]++;
		}
	}

	return 0;
}

int compSS::findVariables() {
	int i1, i2, ic;

	mPosGap = (__m128i*)_aligned_malloc(sizeof(__m128i) * nRows, sizeof(__m128i));
	mPosOne = (__m128i*)_aligned_malloc(sizeof(__m128i) * nRows, sizeof(__m128i));
	vPositionGap = (int*)malloc(sizeof(int) * seedLen);
	vPositionOne = (int*)malloc(sizeof(int) * seedLen);

	vIndGapReverse = (int*)malloc(sizeof(int) * n32 * 32);
	vIndOneReverse = (int*)malloc(sizeof(int) * n32 * 32);

	for (int i = 0; i < n32 * 32; i++) {
		vIndGapReverse[i] = -1;
		vIndOneReverse[i] = -1;
	}
	
	nVar = 0;

	for (int i = 0; i < seedWeight; i++) {
		if (vSeed[i] == 0) nVar++;
	}

	//printf("Number of variables: %i\n", nVar);

	//fprintf(fo, "Number of variables: %i\n", nVar);

	nVar128 = nVar / 128;
	if (nVar % 128 > 0)nVar128++;

	for (int i = 0; i < nRows; i++) {
		mPosGap[i] = mZero;
		mPosOne[i] = mZero;
	}

	ic = 0;
	for (int i = 0; i < seedWeight; i++) {
		i1 = i >> 5;
		i2 = i & 31;
		if (vSeed[i] == 1) continue;
		vIndGapReverse[i] = ic;
		vPositionGap[ic] = i;
		mPosGap[i1] = _mm_or_si128(mPosGap[i1], mOne[i2]);
		ic++;
	}

	ic = 0;
	for (int i = seedWeight; i < seedLen; i++) {
		i1 = i >> 5;
		i2 = i & 31;
		if (vSeed[i] == 0) continue;
		vIndOneReverse[i] = ic;
		vPositionOne[ic] = i;
		mPosOne[i1] = _mm_or_si128(mPosOne[i1], mOne[i2]);
		ic++;
	}
	return 0;
}

int compSS::setM() {
	mZero = _mm_set1_epi32(0);
	for (int i = 0; i < 32; i++) {
		mShift32[i] = _mm_set_epi32(0, 0, 0, i);
		mOne[i] = _mm_set_epi32(0, 0, 0, 1 << i);
		mOne[32 + i] = _mm_set_epi32(0, 0, 1 << i, 0);
		mOne[64 + i] = _mm_set_epi32(0, 1 << i, 0, 0);
		mOne[96 + i] = _mm_set_epi32(1 << i, 0, 0, 0);
	}
	return 0;
}

int compSS::findSeedLen() {
	seedLen = 0;
	
	for (int j = 31; j >= 0; j--) {
		if (vSeed[32 * (nRows - 1) + j] == 0)continue;
		seedLen = 32 * (nRows - 1) + j + 1;
		break;
	}
	//printf("Seed len: %i\n", seedLen);
	//fprintf(fo, "Seed len: %i\n", seedLen);

	n32 = seedLen >> 5;
	if ((seedLen & 31) != 0) n32++;

	return 0;
}

int compSS::prepareSeed() {
	//fprintf(fo, "Seed\n\n");
	for (int i = 0; i < nRows; i++) {
		//printf("%i\t", vIndexRows[i]);
		for (int j = 0; j < 32; j++) {
			vSeed[32 * i + j] = vSeedOrig[32 * vIndexRows[i] + j];
			//fprintf(fo, "%1i", vSeed[32 * i + j]);
		}
		//fprintf(fo, "\n");
	}
	//fprintf(fo, "\n");
	//printf("\n");
	return 0;
}

int main(int argc, char** argv) {
	globalCompass* gc;
	int ires;

	gc = new globalCompass();
	ires = gc->startProcessing(argc, argv);
	delete gc; gc = nullptr;

printf("RESULT: %i\n", ires);

	return ires;
}
