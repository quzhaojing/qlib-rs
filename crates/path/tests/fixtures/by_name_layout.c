/* Independent SDK layout oracle, test-only; no production query implementation. */
#include <sdkddkver.h>
#undef NTDDI_VERSION
#define NTDDI_VERSION NTDDI_WIN11_ZN
#include <windows.h>
#include <stddef.h>
#include <stdio.h>
#define FIELD(name) printf(" %zu", offsetof(FILE_STAT_BASIC_INFORMATION, name))
int main(void) {
    printf("%zu %zu %d %zu", sizeof(FILE_STAT_BASIC_INFORMATION),
        (size_t)__alignof(FILE_STAT_BASIC_INFORMATION), FileStatBasicByNameInfo,
        sizeof(FILE_INFO_BY_NAME_CLASS));
    FIELD(FileId); FIELD(CreationTime); FIELD(LastAccessTime); FIELD(LastWriteTime);
    FIELD(ChangeTime); FIELD(AllocationSize); FIELD(EndOfFile); FIELD(FileAttributes);
    FIELD(ReparseTag); FIELD(NumberOfLinks); FIELD(DeviceType); FIELD(DeviceCharacteristics);
    FIELD(Reserved); FIELD(VolumeSerialNumber); FIELD(FileId128);
    return 0;
}
