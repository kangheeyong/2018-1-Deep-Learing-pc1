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
    FILE *fd1;

    fd = fopen(argv[1],"r");
    if (fd == NULL)
    {
        printf("file read fail\n");
        return 0;
    }
    else
    {

        fd1 = fopen(strtok(argv[1],"."),"wb");

        if (fd1 == NULL)
        {
            printf("file1 load fail\n");
            return 0;
        }
 
        unsigned int size = 1024*1024*10; // 약 10Mbyte
        char *buff = new char[size];
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

        //printf("%d\n",cnt);

        //fscanf( fd, "%s\n", buff);
        //printf( "%s",buff);
        int temp;
        while(fgets( buff, size, fd ))
        {
            //printf( "%s", buff );
            cnt = 0;
            char *token = strtok(buff, ",");
            token = strtok(NULL, ",");
            do
            {
                temp = atoi(token);
                fwrite( &temp, sizeof(int), 1, fd1);
                //fprintf(fd1,"%d", atoi(token));
                cnt++;
                if(cnt == 3648) break;
            }
            while (token = strtok(NULL, ","));
       //     fprintf(fd1,"\n");
         
        }

        delete buff;
        fclose(fd);
        fclose(fd1);
    }


    return 0;
}
