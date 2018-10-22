#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <stdlib.h>


int main(int argc, char** argv)
{
    if(argc != 2)
    {
        printf("USAGE : %s [the data path]\n",argv[0]);
        return 0;
    }

    FILE *fd;
    

    fd = fopen(argv[1],"r");
    if (fd == NULL)
    {
        printf("file read fail\n");
        return 0;
    }
 
    
    char buff[4096];
    int n_size;
    
    /*
    fgets( buff, sizeof(buff), fd );
    printf( "%s", buff );


*/
    int cnt = 0;
    while( 0 < (n_size = fread( buff, sizeof(unsigned char), 1, fd)))
    {
        if(buff[0] == '\n')
        {
            cnt++;
            if(cnt == 3) break;
        }
    }

    printf("%d\n",cnt);

    fgets( buff, sizeof(buff), fd );
    printf( "%s", buff );


    
    fclose(fd);




    return 0;
}
