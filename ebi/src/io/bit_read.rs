/// A trait for reading individual bits, bytes, or sequences of bits from a buffer.
///
/// The `BitRead` trait provides methods for reading binary data at the bit level, allowing
/// precise control over the data being read. This is useful in applications such as data
/// compression, serialization, and communication protocols where bit-level operations are required.
pub trait BitRead2 {
    /// Reads a single bit from the buffer.
    ///
    /// # Returns
    ///
    /// - `Some(true)` if the bit read is 1, `Some(false)` if the bit read is 0, and `None` if the end of the buffer is reached.
    ///
    /// # Note
    ///
    /// - `None` indicates that the end of the buffer has been reached, and no more bits are available for reading.
    fn read_bit(&mut self) -> Option<bool>;

    /// Reads a single byte (8 bits) from the buffer.
    ///
    /// # Returns
    ///
    /// - `Some(u8)` containing the byte value read from the buffer, or `None` if the end of the buffer is reached.
    ///
    /// # Behavior
    ///
    /// This method reads the next 8 bits from the buffer, regardless of the current bit position.
    ///
    /// # Note
    ///
    /// - `None` indicates that the end of the buffer has been reached, and no more bytes are available for reading.
    fn read_byte(&mut self) -> Option<u8>;

    /// Reads the next `n` bits from the buffer.
    ///
    /// # Parameters
    ///
    /// - `n`: The number of bits to read. This value must be between 1 and 64.
    ///
    /// # Returns
    ///
    /// - `Some(u64)` containing the read bits, with the most significant bit of the result
    ///   corresponding to the first bit read from the buffer, or `None` if the end of the buffer is reached before reading all `n` bits.
    ///
    /// # Note
    ///
    /// - `None` indicates that the end of the buffer was reached before the requested number of bits could be read.
    fn read_bits(&mut self, n: u8) -> Option<u64>;

    /// Peeks at the next `n` bits from the buffer without advancing the cursor.
    ///
    /// # Parameters
    ///
    /// - `n`: The number of bits to peek. This value must be between 1 and 64.
    ///
    /// # Returns
    ///
    /// - `Some(u64)` containing the peeked bits, with the most significant bit of the result
    ///   corresponding to the first bit in the peeked range, or `None` if the end of the buffer is reached before peeking all `n` bits.
    ///
    /// # Behavior
    ///
    /// This method allows you to look ahead at the next `n` bits without affecting the current
    /// read position. The cursor remains unchanged after this operation.
    ///
    /// # Note
    ///
    /// - `None` indicates that the end of the buffer was reached before the requested number of bits could be peeked.
    fn peak_bits(&mut self, n: u8) -> Option<u64>;
}

impl<T: BitRead2> BitRead2 for &mut T {
    fn read_bit(&mut self) -> Option<bool> {
        (*self).read_bit()
    }

    fn read_byte(&mut self) -> Option<u8> {
        (*self).read_byte()
    }

    fn read_bits(&mut self, n: u8) -> Option<u64> {
        (*self).read_bits(n)
    }

    fn peak_bits(&mut self, n: u8) -> Option<u64> {
        (*self).peak_bits(n)
    }
}

/// A buffered bit reader that implements the `BitRead` trait.
///
/// `BufferedBitReader` allows reading individual bits, bytes, or sequences of bits
/// from an internal buffer with fine-grained control over the bit-level operations.
///
/// The buffer can be any type that implements `AsRef<[u8]>`, such as `Vec<u8>`, `&[u8]`, or `Box<[u8]>`.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct BufferedBitReader<T: AsRef<[u8]>> {
    buffer: T,
    bit_pos: usize,
    byte_pos: usize,
}

impl<T: AsRef<[u8]>> BufferedBitReader<T> {
    /// Creates a new `BufferedBitReader` from a given buffer.
    ///
    /// # Parameters
    ///
    /// - `buffer`: A buffer of bytes from which the bits will be read. The buffer type can be any type
    ///   that implements `AsRef<[u8]>`, such as `Vec<u8>`, `&[u8]`, or `Box<[u8]>`.
    ///
    /// # Example
    ///
    /// ```
    /// use ebi::io::bit_read::BufferedBitReader;
    ///
    /// let reader = BufferedBitReader::new(vec![0b10101010]);
    /// ```
    pub fn new(buffer: T) -> Self {
        Self {
            buffer,
            bit_pos: 0,
            byte_pos: 0,
        }
    }

    pub fn reset(&mut self) {
        self.bit_pos = 0;
        self.byte_pos = 0;
    }

    pub fn read_bit_unchecked(&mut self) -> bool {
        let buffer = self.buffer.as_ref();
        let bit = (buffer[self.byte_pos] >> (7 - self.bit_pos)) & 1 != 0;
        self.bit_pos += 1;

        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }

        bit
    }

    pub fn read_byte_unchecked(&mut self) -> u8 {
        let buffer = self.buffer.as_ref();
        if self.bit_pos == 0 {
            let byte = buffer[self.byte_pos];
            self.byte_pos += 1;
            byte
        } else {
            let high_bits = buffer[self.byte_pos] << self.bit_pos;
            self.byte_pos += 1;
            let low_bits = if self.byte_pos < buffer.len() {
                buffer[self.byte_pos] >> (8 - self.bit_pos)
            } else {
                0
            };
            high_bits | low_bits
        }
    }

    /// Skips the next `n` bits in the buffer without reading them.
    ///
    /// # Example
    /// ```
    /// use ebi::io::bit_read::BufferedBitReader;
    /// use ebi::io::bit_read::BitRead2 as _;
    ///
    /// let mut reader = BufferedBitReader::new(vec![0b1010_1010, 0b1111_0010, 0b1111_1111, 0b0011_1100]);
    ///
    /// reader.skip_bits(4); // Skips the first 4 bits
    /// assert_eq!(reader.read_bits(3).unwrap(), 0b101); // Reads the next 3 bits
    ///
    /// reader.skip_bits(5); // Skips the next 5 bits
    /// assert_eq!(reader.read_bits(3).unwrap(), 0b001); // Reads the next 3 bits
    ///
    /// reader.skip_bits(9); // Skips the next 9 bits
    /// assert_eq!(reader.read_byte().unwrap(), 0b0011_1100); // Reads the next byte
    ///
    pub fn skip_bits(&mut self, n: usize) {
        let skip_bytes = (n + self.bit_pos) / 8;
        self.byte_pos += skip_bytes;
        let skip_bits = (n + self.bit_pos) % 8;
        self.bit_pos = skip_bits;
    }
}

impl<T: AsRef<[u8]>> BitRead2 for BufferedBitReader<T> {
    /// Reads a single bit from the buffer.
    ///
    /// # Returns
    ///
    /// - `Some(true)` if the bit read is 1, `Some(false)` if the bit read is 0, and `None` if the end of the buffer is reached.
    ///
    /// # Note
    ///
    /// - `None` indicates that the end of the buffer has been reached, and no more bits are available for reading.
    ///
    /// # Example
    ///
    /// ```
    /// use ebi::io::bit_read::BufferedBitReader;
    /// use ebi::io::bit_read::BitRead2 as _;
    ///
    /// let mut reader = BufferedBitReader::new(vec![0b10101010]);
    /// let bit = reader.read_bit().unwrap();
    /// assert_eq!(bit, true); // Reads the first bit (1)
    ///
    /// reader.read_bits(7).unwrap(); // Reads the remaining 7 bits
    /// assert_eq!(reader.read_bit(), None); // End of buffer
    /// ```
    fn read_bit(&mut self) -> Option<bool> {
        let buffer = self.buffer.as_ref();
        if self.byte_pos >= buffer.len() {
            return None;
        }

        Some(self.read_bit_unchecked())
    }

    /// Reads a single byte (8 bits) from the buffer.
    ///
    /// # Returns
    ///
    /// - `Some(u8)` containing the byte value read from the buffer, or `None` if the end of the buffer is reached.
    ///
    /// # Behavior
    ///
    /// This method reads the next 8 bits from the buffer, regardless of the current bit position.
    ///
    /// # Note
    ///
    /// - `None` indicates that the end of the buffer has been reached, and no more bytes are available for reading.
    ///
    /// # Example
    ///
    /// ```
    /// use ebi::io::bit_read::BufferedBitReader;
    /// use ebi::io::bit_read::BitRead2 as _;
    ///
    /// let mut reader = BufferedBitReader::new(vec![0xAA, 0b0101_1111, 0b1100_0011]);
    /// let byte = reader.read_byte().unwrap();
    /// assert_eq!(byte, 0xAA);
    /// assert_eq!(reader.read_bit().unwrap(), false);
    ///
    /// // Unaligned byte read
    /// let byte = reader.read_byte().unwrap();
    /// assert_eq!(byte, 0b1011_1111);
    ///
    /// assert_eq!(reader.read_byte(), None); // End of buffer
    /// ```
    fn read_byte(&mut self) -> Option<u8> {
        let buffer = self.buffer.as_ref();
        if self.byte_pos >= buffer.len() {
            return None;
        }
        if self.byte_pos + 1 >= buffer.len() && self.bit_pos != 0 {
            return None;
        }

        Some(self.read_byte_unchecked())
    }

    /// Reads the next `n` bits from the buffer.
    ///
    /// # Parameters
    ///
    /// - `n`: The number of bits to read. This value must be between 1 and 64.
    ///
    /// # Returns
    ///
    /// - `Some(u64)` containing the read bits, with the most significant bit of the result
    ///   corresponding to the first bit read from the buffer, or `None` if the end of the buffer is reached before reading all `n` bits.
    ///
    /// # Note
    ///
    /// - `None` indicates that the end of the buffer was reached before the requested number of bits could be read.
    ///
    /// # Example
    ///
    /// ```
    /// use ebi::io::bit_read::BufferedBitReader;
    /// use ebi::io::bit_read::BitRead2 as _;
    ///
    /// let mut reader = BufferedBitReader::new(vec![0b10110000, 0b11110000]);
    /// let bits = reader.read_bits(4).unwrap();
    /// assert_eq!(bits, 0b1011);
    ///
    /// let bits = reader.read_bits(12).unwrap();
    /// assert_eq!(bits, 0b0000_1111_0000);
    ///
    /// assert_eq!(reader.read_bits(4), None); // End of buffer
    /// ```
    fn read_bits(&mut self, n: u8) -> Option<u64> {
        let mut remaining_bits = n;
        let mut result = 0u64;

        if (self.buffer.as_ref().len() - self.byte_pos) * 8 - self.bit_pos < n as usize {
            return None;
        }

        // Read full bytes if possible
        while remaining_bits >= 8 {
            let byte = self.read_byte_unchecked();
            result = (result << 8) | (byte as u64);
            remaining_bits -= 8;
        }

        // Read remaining bits one by one
        for _ in 0..remaining_bits {
            result <<= 1;
            let bit = self.read_bit_unchecked();
            if bit {
                result |= 1;
            }
        }

        Some(result)
    }

    /// Peeks at the next `n` bits from the buffer without advancing the cursor.
    ///
    /// # Parameters
    ///
    /// - `n`: The number of bits to peek. This value must be between 1 and 64.
    ///
    /// # Returns
    ///
    /// - `Some(u64)` containing the peeked bits, with the most significant bit of the result
    ///   corresponding to the first bit in the peeked range, or `None` if the end of the buffer is reached before peeking all `n` bits.
    ///
    /// # Behavior
    ///
    /// This method allows you to look ahead at the next `n` bits without affecting the current
    /// read position. The cursor remains unchanged after this operation.
    ///
    /// # Note
    ///
    /// - `None` indicates that the end of the buffer was reached before the requested number of bits could be peeked.
    ///
    /// # Example
    ///
    /// ```
    /// use ebi::io::bit_read::BufferedBitReader;
    /// use ebi::io::bit_read::BitRead2 as _;
    ///
    /// let mut reader = BufferedBitReader::new(vec![0b10110000]);
    /// let bits = reader.peak_bits(4).unwrap();
    /// assert_eq!(bits, 0b1011);
    /// // The cursor has not advanced, so the next read will still read from the start
    /// let bits_after_peek = reader.read_bits(4).unwrap();
    /// assert_eq!(bits_after_peek, 0b1011);
    ///
    /// assert_eq!(reader.peak_bits(4), Some(0b0000));
    /// assert_eq!(reader.peak_bits(5), None); // End of buffer
    /// ```
    fn peak_bits(&mut self, n: u8) -> Option<u64> {
        let original_byte_pos = self.byte_pos;
        let original_bit_pos = self.bit_pos;

        let bits = self.read_bits(n);

        // Restore the original cursor positions
        self.byte_pos = original_byte_pos;
        self.bit_pos = original_bit_pos;

        bits
    }
}
