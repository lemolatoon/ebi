use std::{
    io::{self, BufRead, Cursor, Read, Seek, Write},
    mem::size_of_val,
};

use super::aligned_buf_reader::AlignedBufRead;

pub struct AlignedCursor<T> {
    cursor: Cursor<T>,
}

impl<'a> AlignedCursor<&'a [u8]> {
    pub fn try_new(data: &'a [u8]) -> Option<Self> {
        if !data.as_ptr().cast::<u64>().is_aligned() {
            return None;
        }
        Some(Self {
            cursor: Cursor::new(data),
        })
    }

    pub fn from_vec<U>(data: &'a Vec<U>) -> Self {
        const {
            assert!(
                std::mem::align_of::<U>() % 8 == 0,
                "T must be 8-byte aligned"
            );
        };
        let slice: &'a [U] = data.as_slice();
        // Safety:
        // `slice` is 8-byte aligned.
        // `slice` will live as long as `data` whose lifetime is `'a`.
        // Any bit representations of the byte slice are valid.
        let u8_slice: &'a [u8] =
            unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), size_of_val(slice)) };
        Self {
            cursor: Cursor::new(u8_slice),
        }
    }
}

impl<'a> AlignedCursor<&'a mut [u8]> {
    pub fn try_new(data: &'a mut [u8]) -> Option<Self> {
        if !data.as_ptr().cast::<u64>().is_aligned() {
            return None;
        }
        Some(Self {
            cursor: Cursor::new(data),
        })
    }

    pub fn from_vec_mut<U>(data: &'a mut Vec<U>) -> Self {
        const {
            assert!(
                std::mem::align_of::<U>() % 8 == 0,
                "T must be 8-byte aligned"
            );
        };
        let slice: &'a mut [U] = data.as_mut_slice();
        // Safety:
        // `slice` is 8-byte aligned.
        // `slice` will live as long as `data` whose lifetime is `'a`.
        // Any bit representations of the byte slice are valid.
        let u8_slice: &'a mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<u8>(), size_of_val(slice))
        };
        Self {
            cursor: Cursor::new(u8_slice),
        }
    }
}

impl<T> AlignedCursor<T> {
    pub fn into_inner(self) -> Cursor<T> {
        self.cursor
    }
}

impl<T> Read for AlignedCursor<T>
where
    Cursor<T>: Read,
{
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.cursor.read(buf)
    }
}

impl<T> BufRead for AlignedCursor<T>
where
    Cursor<T>: BufRead,
{
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.cursor.fill_buf()
    }

    fn consume(&mut self, amt: usize) {
        self.cursor.consume(amt)
    }
}

// Safety:
// `AlignedCursor`'s internal buffer is 8-byte aligned.
unsafe impl<T> AlignedBufRead for AlignedCursor<T>
where
    Cursor<T>: BufRead,
{
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        BufRead::fill_buf(self)
    }

    fn consume(&mut self, amt: usize) {
        BufRead::consume(self, amt)
    }
}

impl<T> Write for AlignedCursor<T>
where
    Cursor<T>: Write,
{
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.cursor.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.cursor.flush()
    }
}

impl<T> Seek for AlignedCursor<T>
where
    Cursor<T>: Seek,
{
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        self.cursor.seek(pos)
    }
}
