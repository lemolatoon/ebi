use tsz::{stream, Bit};

/// BufferedWriter
///
/// BufferedWriter writes bytes to a buffer.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct BufferedWriterExt {
    buf: Vec<u8>,
    pos: u32, // position in the last byte in the buffer
}

impl Default for BufferedWriterExt {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferedWriterExt {
    /// new creates a new BufferedWriter
    pub fn new() -> Self {
        Self::from_vec(Vec::new())
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::from_vec(Vec::with_capacity(capacity))
    }

    /// from_vec creates a new BufferedWriter from a Vec<u8>
    /// This is useful when you want to reuse a buffer
    /// The buffer's data will be cleared before initializing BufferedWriterExt,
    /// but the capacity will remain the same
    pub fn from_vec(mut buf: Vec<u8>) -> Self {
        buf.clear();

        Self {
            buf,

            // set pos to 8 to indicate the buffer has no space presently since it is empty
            pos: 8,
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.buf
    }

    pub fn reset(&mut self) {
        self.buf.clear();
        self.pos = 8;
    }

    fn grow(&mut self) {
        self.buf.push(0);
    }

    fn last_index(&self) -> usize {
        self.buf.len() - 1
    }
}

impl stream::Write for BufferedWriterExt {
    fn write_bit(&mut self, bit: Bit) {
        if self.pos == 8 {
            self.grow();
            self.pos = 0;
        }

        let i = self.last_index();

        match bit {
            Bit::Zero => (),
            Bit::One => self.buf[i] |= 1u8.wrapping_shl(7 - self.pos),
        };

        self.pos += 1;
    }

    fn write_byte(&mut self, byte: u8) {
        if self.pos == 8 {
            self.grow();

            let i = self.last_index();
            self.buf[i] = byte;
            return;
        }

        let i = self.last_index();
        let mut b = byte.wrapping_shr(self.pos);
        self.buf[i] |= b;

        self.grow();

        b = byte.wrapping_shl(8 - self.pos);
        self.buf[i + 1] |= b;
    }

    fn write_bits(&mut self, mut bits: u64, mut num: u32) {
        // we should never write more than 64 bits for a u64
        if num > 64 {
            num = 64;
        }

        bits = bits.wrapping_shl(64 - num);
        while num >= 8 {
            let byte = bits.wrapping_shr(56);
            self.write_byte(byte as u8);

            bits = bits.wrapping_shl(8);
            num -= 8;
        }

        while num > 0 {
            let byte = bits.wrapping_shr(63);
            if byte == 1 {
                self.write_bit(Bit::One);
            } else {
                self.write_bit(Bit::Zero);
            }

            bits = bits.wrapping_shl(1);
            num -= 1;
        }
    }

    fn close(self) -> Box<[u8]> {
        self.buf.into_boxed_slice()
    }
}
