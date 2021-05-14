#pragma once

#include <windows.h>

class FileGetter {
	WIN32_FIND_DATAA found;
	HANDLE hfind;
	char folder[MAX_PATH];
	int chk;
	bool first;
	bool hasFiles;
public:
	FileGetter(char* folderin, char* ext);
	int getNextFile(char* fname);
	int getNextAbsFile(char* fname);
	char* getFoundFileName();
};


int openFileDlg(char* fname);

int openFolderDlg(char* folderName);
